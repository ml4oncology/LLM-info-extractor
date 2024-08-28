import argparse
import logging
import os
import warnings
from pathlib import Path

import datetime
import pandas as pd
import numpy as np
import yaml

from .train import get_trainer, time_series_cross_validate
from .util import (
    get_manually_labeled_data, 
    get_label_count, 
    get_predictions, 
    load_tokenizer,
    prepare_dataset, 
)

# User warning occurs for multi-GPU by torch
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
# User warning occurs when using deepspeed
warnings.filterwarnings("ignore", message="UserWarning: Positional args are being deprecated, use kwargs instead.")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S',
)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))
ROOT_DIR = Path(__file__).parent.parent.as_posix()
logger.addHandler(logging.FileHandler(f'{ROOT_DIR}/results/logs/{datetime.datetime.now()}.log', mode='w'))


def parse_args():
    parser = argparse.ArgumentParser()
    # File paths
    parser.add_argument('--csv', type=str, help='Path to the csv file')
    parser.add_argument('--config', type=str, help='Path to the config yaml file')

    # Data params
    parser.add_argument('--label-col', type=str, help='Label column name')
    parser.add_argument('--patient-col', type=str, default='patientid', help='Patient column name')
    parser.add_argument('--date-col', type=str, default='datetime', help='Datetime column name')
    parser.add_argument('--text-col', type=str, default='text', help='Report text column name')

    # Model params
    msg = 'Path to the pre-trained large language model to fine-tune. Must be supported by Huggingface'
    parser.add_argument('--model', type=str, default='gpt2', help=msg)
    msg = 'Quantize the model and fine-tune it via low-rank adaptation (LoRA)'
    parser.add_argument('--lora-quantize', action='store_true', help=msg)
    msg = 'Spread the model across available GPUs, CPUs, and even disk'
    parser.add_argument('--balance', action='store_true', help=msg)
    parser.add_argument('--resume-training', action='store_true', help='Resume training from a checkpoint')

    # Output params
    msg = 'Writes over the original csv to include a new column with the label predictions'
    parser.add_argument('--overwrite-file', action='store_true', help=msg)
    parser.add_argument('--output-path', type=str)

    args = parser.parse_args()
    return vars(args)

def main():
    cfg = parse_args()
    LABEL = cfg['label_col']
    PATIENT = cfg['patient_col']
    DATE = cfg['date_col']
    TEXT = cfg['text_col']
    MODEL_NAME = cfg['model_name'] = Path(cfg['model']).name
    
    # Load data
    csv_path, config_path = cfg.pop('csv'), cfg.pop('config')
    with open(config_path) as file:
        cfg.update(yaml.safe_load(file))
    full_dataset = pd.read_csv(csv_path, parse_dates=[DATE])
    labeled_dataset = get_manually_labeled_data(full_dataset, label_col=LABEL, patient_col=PATIENT, verbose=True)
    
    def show_note_size(x):
        return pd.concat([
            x.str.len().describe(),
            x.str.split().str.len().describe(),
        ], keys=['length of text', 'number of words'], axis=1)
    print(show_note_size(labeled_dataset[TEXT]))

    # Split the data into development and testing set
    year = labeled_dataset[DATE].dt.year
    dev_mask = year.between(cfg['dev_start_year'], cfg['test_start_year'], inclusive='left')
    test_mask = year.between(cfg['test_start_year'], cfg['test_end_year'])
    dev_df, test_df = labeled_dataset[dev_mask], labeled_dataset[test_mask]

    if cfg['cross_validation_interval'] is not None:
        # Train and evaluate model using time-series cross-validation on the development set
        scores = time_series_cross_validate(dev_df, cfg)

    # Train the model using the entire development set
    trainer = get_trainer(dev_df, test_df, cfg)
    trainer.train(resume_from_checkpoint=cfg['resume_training'])
    trainer.save_model(f'{ROOT_DIR}/models/fine-tuned-{MODEL_NAME}-{LABEL}')
    # save the log history
    log_hist = pd.DataFrame(trainer.state.log_history)
    log_hist.to_csv(f'{ROOT_DIR}/results/{MODEL_NAME}-{LABEL}/log_history.csv')
    # Evaluate on the test set
    eval_scores = trainer.evaluate()
    test_set_name = f'Test Set: {test_df[DATE].min().date()} - {test_df[DATE].max().date()}'
    scores = pd.DataFrame({test_set_name: eval_scores}).T
    scores.to_csv(f'{ROOT_DIR}/results/{MODEL_NAME}-{LABEL}/final_evaluation_scores.csv')
    logger.info(f"{test_set_name}\n{get_label_count(test_df, LABEL, PATIENT)}\n")
    logger.info(f'Final evaluation scores:\n{scores.to_string()}')

    # Use the final model to label the unlabeled data
    N = len(full_dataset)
    dummy_labels = pd.Series(np.random.randint(2, size=N))
    tokenizer = load_tokenizer(cfg['model'])
    data = prepare_dataset(full_dataset[TEXT], dummy_labels, tokenizer)
    logits, label_ids, metrics = trainer.predict(data)
    pred_bool, pred_prob = get_predictions(logits)
    if cfg['overwrite_file']:
        output_path = csv_path
        output = full_dataset
    else:
        output_path = cfg.get('output_path', csv_path.replace('.csv', '_labels.csv'))
        output = pd.read_csv('output_path') if os.path.exists(output_path) else full_dataset[[PATIENT, DATE]]
    output[f'{MODEL_NAME}_pred_{LABEL}'] = pred_bool
    output[f'{MODEL_NAME}_pred_prob_{LABEL}'] = pred_prob
    output.to_csv(output_path, index=False)

    
if __name__ == '__main__':
    """
    How to run the script example:

    > python scripts/labeler.py \
        --csv ./data/DVT.csv \
        --config ./config/fine_tune_config.yaml \
        --label-col DVT
        --model ./models/gpt2
        --overwrite-file

    TODO: try without huggingface Trainer (normal PyTorch training, with DDP, model parallelization, etc)
    TODO: Consolidate all data into one excel sheet (time series results, final results, number of counts)
    TODO: include 95% CI 
    """
    main()
