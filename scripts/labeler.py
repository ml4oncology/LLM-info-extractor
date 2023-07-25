import argparse
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.as_posix()
sys.path.append(ROOT_DIR)

from sklearn.model_selection import StratifiedKFold
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments, Trainer
)
import pandas as pd
import numpy as np

from src import logger
from src.util import (
    get_manually_labeled_data,
    get_pretrained_model,
    prepare_dataset,
    compute_metrics
)

import warnings
# User warning occurs for multi-GPU by torch
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to the csv file')
    parser.add_argument('--text-column', type=str, default='text')
    parser.add_argument('--label-column', type=str, default='label')
    parser.add_argument('--patient-column', type=str, default='mrn')
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
    filepath = args.csv
    model_name = args.model_name
    text_col = args.text_column
    label_col = args.label_column
    patient_col = args.patient_column
    overwrite_file = args.overwrite_file
    
    # Load data
    df = pd.read_csv(filepath)
    X, Y = get_manually_labeled_data(
        df=df, 
        text_col=text_col, 
        label_col=label_col, 
        patient_col=patient_col
    )

    # Create training arguments
    training_args = TrainingArguments(
        num_train_epochs=50,              # total # of training epochs
        per_device_train_batch_size=4,    # batch size per device during training
        per_device_eval_batch_size=4,     # batch size per device during evaluation

        # save memory by using 8-bit Adam optimizer...didn't make much difference
        # optim="adamw_bnb_8bit",
        optim="adamw_torch",

        # save memory by imitating large batch sizes 
        # (e.g. batch size x gradient accumulation step = effective batch size)
        gradient_accumulation_steps=4,   # number of times to accumulate gradients before updating weights

        # save memory (at the expense of slower computation) by storing only select activations from forward 
        # pass and recompute the unstored activations during backward pass
        # gradient_checkpointing=True,

        # save memory (and speed up computation) by using floating point 16-bit precision instead of 32-bit
        fp16=True,

        load_best_model_at_end=True,
        metric_for_best_model='AUROC',
        save_strategy='epoch',            # save model checkpoint at end of each epoch
        save_total_limit=1,               # max number of model checkpoints to keep on disk

        evaluation_strategy='epoch',      # evaluate at end of each epoch
        logging_strategy='epoch',         # log at end of each epoch

        output_dir=f'{ROOT_DIR}/results', # output directory
        overwrite_output_dir=True,
        seed=42,
    )
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=5)
    kwargs = {'args': training_args, 'compute_metrics': compute_metrics, 'callbacks': [early_stop_callback]}

    # Train and evaluate model through cross-validation
    scores = {}
    n_splits = 4 # 75-25 split
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for fold, (train_idxs, test_idxs) in enumerate(kf.split(X, Y)):
        logger.info('#############################################')
        logger.info(f'# FOLD {fold+1}')
        logger.info('#############################################')

        tokenizer, model = get_pretrained_model(model_name=model_name)
        # NOTE: max_length: gpt2 = 512
        tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        train_data = prepare_dataset(X, Y, train_idxs, tokenize_function)
        test_data = prepare_dataset(X, Y, test_idxs, tokenize_function)
        
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )
        train_result = trainer.train()
        eval_scores = trainer.evaluate()
        scores[f'Fold {fold+1}'] = eval_scores
    scores = pd.DataFrame(scores).T
    logger.info('Cross validation evaluation scores:')
    logger.info(f'\n{scores}')
    scores.to_csv(f'{ROOT_DIR}/results/{model_name}_evaluation_scores.csv')
    scores = pd.read_csv(f'{ROOT_DIR}/results/{model_name}_evaluation_scores.csv')
    
    # Train the final model using all samples
    logger.info('Training final model using all labeled samples...')
    training_args.num_train_epochs = int(scores['epoch'].mean()) # prevent overfitting...
    tokenizer, model = get_pretrained_model(model_name=model_name)
    tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data = prepare_dataset(X, Y, range(len(X)), tokenize_function)
    trainer = Trainer(
        model=model,
        train_dataset=data,
        eval_dataset=data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **kwargs
    )
    trainer.train()
    trainer.save_model(f'{ROOT_DIR}/models/fine-tuned-{model_name}')

    # Use the final model to label the unlabeled data
    # model = AutoModelForSequenceClassification.from_pretrained(f'{ROOT_DIR}/models/fine-tuned-{model_name}')
    # tokenizer = AutoTokenizer.from_pretrained(f'{ROOT_DIR}/models/fine-tuned-{model_name}')
    # tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
    # trainer = Trainer(model=model)
    N = len(df)
    dummy_labels = pd.Series(np.random.randint(2, size=N))
    data = prepare_dataset(df[text_col], dummy_labels, range(N), tokenize_function)
    predictions, label_ids, metrics = trainer.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    df[f'predicted_{label_col}'] = predicted_labels
    if not overwrite_file:
        filename = filepath.split('/')[-1]
        filepath = filepath.replace(filename, f'{model_name}-{filename}')
    df.to_csv(filepath, index=False)
    
if __name__ == '__main__':
    """
    How to run the script:

    > torchrun scripts/labeler.py \
        --csv data/LabeledVTE-Doppler.csv \
        --text-column text \
        --label-column DVT \
        --patient-column patientid
        --model-name gpt2
        --overwrite-file

    TODO: try deepspeed tutorial
    TODO: try without huggingface Trainer (normal PyTorch training, with DDP, model parallelization, etc)
    TODO: try gptneo and bert
    """
    main()
