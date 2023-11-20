import argparse
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.as_posix()
sys.path.append(ROOT_DIR)

from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments, Trainer
)
import datetime
import pandas as pd
import numpy as np
import yaml

from src import logger
from src.util import (
    get_manually_labeled_data,
    get_pretrained_model,
    prepare_dataset,
    compute_metrics,
    get_label_count
)

import logging
import warnings
# User warning occurs for multi-GPU by torch
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
# User warning occurs when using deepspeed
warnings.filterwarnings("ignore", message="UserWarning: Positional args are being deprecated, use kwargs instead.")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler(f'{ROOT_DIR}/results/logs/{datetime.datetime.now()}.log', mode='w'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to the csv file')
    parser.add_argument('--config', type=str, help='Path to the config yaml file')
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='gpt2',
        help='Name of the large language model to fine-tune. Must be supported by Huggingface'
    )
    parser.add_argument(
        '--overwrite-file', 
        action='store_true', 
        help='If True, writes over the original csv to include a new column with the label predictions'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    csv_path = args.csv
    config_path = args.config
    model_name = args.model_name
    overwrite_file = args.overwrite_file
    
    # Load data
    with open(config_path) as file:
        cfg = yaml.safe_load(file)
    LABEL, PATIENT, DATE, TEXT = cfg['label_col'], cfg['patient_col'], cfg['date_col'], cfg['text_col']
    df = pd.read_csv(csv_path, parse_dates=[DATE])
    df = get_manually_labeled_data(df, label_col=LABEL, patient_col=PATIENT)

    # Create training arguments
    training_args = TrainingArguments(**cfg['training_args'])
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=cfg['early_stopping_patience'])
    kwargs = {'args': training_args, 'compute_metrics': compute_metrics, 'callbacks': [early_stop_callback]}

    # Split the data into development and testing set
    test_mask = df[DATE].dt.year >= cfg['test_split_year']
    dev_df, test_df = df[~test_mask], df[test_mask]

    # Train and evaluate model using time-series cross-validation on the development set
    # compute the time splits
    start_date = dev_df[DATE].min()
    time_splits = pd.date_range(
        pd.Timestamp(year=cfg['dev_start_year'], month=1, day=1), 
        pd.Timestamp(year=cfg['test_split_year'], month=1, day=1),
        freq=f"{cfg['interval']}M", 
        closed='left' # inclusive='left'
    )
    time_splits = time_splits[1:] # discard the first date - no need
    # run the cross validation
    scores = {}
    for i in range(len(time_splits) - 1):
        train_end = time_splits[i]
        valid_end = time_splits[i+1]
        train_df = dev_df[dev_df[DATE] < train_end]
        valid_df = dev_df[dev_df[DATE].between(train_end, valid_end)]

        logger.info('#######################################################')
        train_fold_name = f'Train Fold: {start_date.date()} - {train_end.date()}'
        valid_fold_name = f'Validation Fold: {train_end.date()} - {valid_end.date()}'
        logger.info(f"{train_fold_name}\n{get_label_count(train_df, LABEL, PATIENT)}\n")
        logger.info(f"{valid_fold_name}\n{get_label_count(valid_df, LABEL, PATIENT)}\n")

        tokenizer, model = get_pretrained_model(model_name=model_name)
        # NOTE: max_length: gpt2 = 512
        tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        train_data = prepare_dataset(train_df[TEXT], train_df[LABEL], tokenize_function)
        valid_data = prepare_dataset(valid_df[TEXT], valid_df[LABEL], tokenize_function)
        
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=valid_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )
        train_result = trainer.train()
        eval_scores = trainer.evaluate()
        scores[f'{train_fold_name}\n{valid_fold_name}'] = eval_scores
    scores = pd.DataFrame(scores).T
    logger.info(f'Cross validation evaluation scores:\n{scores.to_string()}')
    scores.to_csv(f'{ROOT_DIR}/results/{model_name}_{LABEL}_tscv_evaluation_scores.csv')
    scores = pd.read_csv(f'{ROOT_DIR}/results/{model_name}_{LABEL}_tscv_evaluation_scores.csv')
    
    # Train the final model using the entire development set and evaluate on the test set
    # NOTE: why not incorporate the test set into the time-series cross validation? Seems pretty trivial.
    # Cuz I might want to use the time-series cross validation for parameter tuning, which should never use the test set.
    # So to keep it more general for future use case, I separated the two.
    logger.info('Training final model using all labeled samples...')
    training_args.num_train_epochs = int(scores['epoch'].mean()) # prevent overfitting...
    tokenizer, model = get_pretrained_model(model_name=model_name)
    tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dev_data = prepare_dataset(dev_df[TEXT], dev_df[LABEL], tokenize_function)
    test_data = prepare_dataset(test_df[TEXT], test_df[LABEL], tokenize_function)
    trainer = Trainer(
        model=model,
        train_dataset=dev_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **kwargs
    )
    trainer.train()
    trainer.save_model(f'{ROOT_DIR}/models/fine-tuned-{model_name}-{LABEL}')
    eval_scores = trainer.evaluate()
    test_set_name = f'Test Set: {test_df[DATE].min().date()} - {test_df[DATE].max().date()}'
    logger.info(f"{test_set_name}\n{get_label_count(test_df, LABEL, PATIENT)}\n")
    scores = pd.DataFrame({test_set_name: eval_scores}).T
    logger.info(f'Final evaluation scores:\n{scores.to_string()}')
    scores.to_csv(f'{ROOT_DIR}/results/{model_name}_{LABEL}_final_evaluation_scores.csv')

    # Use the final model to label the unlabeled data
    # model = AutoModelForSequenceClassification.from_pretrained(f'{ROOT_DIR}/models/fine-tuned-{model_name}')
    # tokenizer = AutoTokenizer.from_pretrained(f'{ROOT_DIR}/models/fine-tuned-{model_name}')
    # tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
    # trainer = Trainer(model=model)
    N = len(df)
    dummy_labels = pd.Series(np.random.randint(2, size=N))
    data = prepare_dataset(df[TEXT], dummy_labels, tokenize_function)
    predictions, label_ids, metrics = trainer.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    df[f'predicted_{LABEL}'] = predicted_labels
    if not overwrite_file:
        csv_name = csv_path.split('/')[-1]
        csv_path = csv_path.replace(csv_name, f'{model_name}-{csv_name}')
    df.to_csv(csv_path, index=False)
    
if __name__ == '__main__':
    """
    How to run the script example:

    > torchrun scripts/labeler.py \
        --csv data/LabeledVTE-Doppler.csv \
        --config config/config.yaml \
        --model-name gpt2
        --overwrite-file

    TODO: try without huggingface Trainer (normal PyTorch training, with DDP, model parallelization, etc)
    TODO: try gptneo and bert
    """
    main()
