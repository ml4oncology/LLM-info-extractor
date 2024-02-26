from transformers import (
    DataCollatorWithPadding, 
    EarlyStoppingCallback, 
    Trainer, 
    TrainingArguments, 
)
from trl import SFTTrainer
import pandas as pd

from . import ROOT_DIR, logger
from .util import (
    compute_metrics,
    get_label_count,
    get_peft_config,
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


def get_trainer(train_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict, **kwargs) -> Trainer | SFTTrainer:
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
    tokenizer, model = get_pretrained_model(model_path=cfg['model'], quantize=cfg['quantize'])
    # NOTE: max_length: gpt2 = 1024, bert = 512, clinical-longformer = 4096
    tokenize_function = lambda x: tokenizer(x["text"], padding="max_length", truncation=True) # max_length=512
    train_data = prepare_dataset(train_df[TEXT], train_df[LABEL], tokenize_function)
    eval_data = prepare_dataset(eval_df[TEXT], eval_df[LABEL], tokenize_function)
        
    if cfg['adapt']:
        trainer = SFTTrainer
        kwargs['dataset_batch_size'] = training_args.per_device_train_batch_size
        kwargs['dataset_text_field'] = 'text'
        kwargs['peft_config'] = get_peft_config(**cfg['peft_args'])
    else:
        trainer = Trainer
        kwargs['data_collator'] = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return trainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        **kwargs
    )