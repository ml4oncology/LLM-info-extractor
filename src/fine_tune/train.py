from transformers import (
    DataCollatorWithPadding, 
    EarlyStoppingCallback, 
    Trainer, 
    TrainingArguments, 
)
import pandas as pd

from . import ROOT_DIR, logger
from .util import (
    compute_metrics,
    get_label_count,
    get_pretrained_model,
    prepare_dataset,
)

def time_series_cross_validate(df: pd.DataFrame, cfg: dict, **kwargs) -> pd.DataFrame:
    """
    Args:
        **kwargs: keyword arguments fed into get_trainer
    """
    LABEL, PATIENT, DATE = cfg['label_col'], cfg['patient_col'], cfg['date_col']
    kwargs['callbacks'] = [EarlyStoppingCallback(early_stopping_patience=cfg['early_stopping_patience'])]
    
    # compute the time splits
    start_date = df[DATE].min()
    time_splits = pd.date_range(
        pd.Timestamp(year=cfg['dev_start_year'], month=1, day=1), 
        pd.Timestamp(year=cfg['test_split_year'], month=1, day=1),
        freq=f"{cfg['interval']}MS", # MS - Month Start 
    )
    time_splits = time_splits[1:] # discard the first date - no need
    # run the cross validation
    scores = {}
    for i in range(len(time_splits) - 1):
        train_end = time_splits[i]
        valid_end = time_splits[i+1]
        train_df = df[df[DATE] < train_end]
        valid_df = df[df[DATE].between(train_end, valid_end, inclusive='left')]

        train_fold_name = f'Train Fold: {start_date.date()} - {train_end.date()}'
        valid_fold_name = f'Validation Fold: {train_end.date()} - {valid_end.date()}'
        logger.info('#######################################################')
        logger.info(f"{train_fold_name}\n{get_label_count(train_df, LABEL, PATIENT)}\n")
        logger.info(f"{valid_fold_name}\n{get_label_count(valid_df, LABEL, PATIENT)}\n")

        trainer = get_trainer(train_df, valid_df, cfg, **kwargs)
        train_result = trainer.train()
        eval_scores = trainer.evaluate()
        scores[f'{train_fold_name}\n{valid_fold_name}'] = eval_scores
    scores = pd.DataFrame(scores).T
    logger.info(f'Cross validation evaluation scores:\n{scores.to_string()}')
    scores.to_csv(f'{ROOT_DIR}/results/{cfg["model_name"]}-{LABEL}/tscv_evaluation_scores.csv')
    return scores


def get_trainer(train_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict, **kwargs) -> Trainer:
    """
    Args:
        **kwargs: keyword arguments fed into Trainer or SFTTrainer
    """
    LABEL, TEXT, MODEL_NAME = cfg['label_col'], cfg['text_col'], cfg['model_name']

    # Create training arguments
    training_args = TrainingArguments(
        # output directory where the model predictions and checkpoints will be written.
        output_dir=f'{ROOT_DIR}/results/{MODEL_NAME}-{LABEL}',
        overwrite_output_dir=True,
        **cfg['training_args']
    )

    # Get the tokenizer, model, and prepped data
    tokenizer, model = get_pretrained_model(cfg)
    train_data = prepare_dataset(train_df[TEXT], train_df[LABEL], tokenizer)
    eval_data = prepare_dataset(eval_df[TEXT], eval_df[LABEL], tokenizer)
    
    return Trainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        **kwargs
    )