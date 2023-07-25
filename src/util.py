from typing import Callable, Sequence
import os

from datasets import Dataset
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    accuracy_score, recall_score, precision_score, f1_score
)
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    BertConfig, BertTokenizerFast, BertForSequenceClassification,
    GPT2Config, GPT2TokenizerFast, GPT2ForSequenceClassification,
)
import pandas as pd
import numpy as np
import torch

from . import ROOT_DIR, logger

###############################################################################
# Load Model
###############################################################################
def get_pretrained_model(model_name: str, freeze_encoder: bool = False):
    """Load the pretrained tokenizer and model"""
    model_path = f'{ROOT_DIR}/models/{model_name}'
    if not os.path.exists(model_path):
        download_pretrained_model(model_name, model_path)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f'{model_path}/tokenizer')
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        f'{model_path}/model', 
        problem_type='single_label_classification'
    )
    model.config.pad_token_id = model.config.eos_token_id
    logger.info(f'Number of parameters in {model_name.upper()}: {model.num_parameters():,}')
    
    if freeze_encoder:
        # keep about half of the encoder frozen
        for idx, (name, param) in enumerate(model.named_parameters()):
            if idx % 2 == 0 and 'bias' not in name: 
                param.requires_grad = False

    if torch.cuda.is_available():
        model = model.to('cuda')

    return tokenizer, model

def download_pretrained_model(model_name: str, model_path: str):
    """Download and save the pre-trained tokenizer and model"""
    if model_name == 'gpt2':
        Tokenizer = GPT2TokenizerFast
        Config = GPT2Config
        Model = GPT2ForSequenceClassification
    else:
        Tokenizer = AutoTokenizer
        Config = AutoConfig
        Model = AutoModelForSequenceClassification
    
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f'{model_path}/tokenizer')

    model_config = Config.from_pretrained(model_name, num_labels=2) # binary classification
    model = Model.from_pretrained(model_name, config=model_config)
    model.save_pretrained(f'{model_path}/model')

###############################################################################
# Data Prep
###############################################################################
def get_manually_labeled_data(
    df: pd.DataFrame, 
    text_col: str, 
    label_col: str, 
    patient_col: str = 'mrn',
    verbose: bool = True
):
    mask = df[label_col].notnull()
    labeled_df = df[mask].copy()

    if verbose:
        count = pd.concat([
            labeled_df[label_col].value_counts(),
            labeled_df.groupby(label_col).apply(lambda g: g[patient_col].nunique())
        ], axis=1, keys=['Sessions', 'Patients']).sort_index()
        logger.info(f'\n{count}')

    X, Y = labeled_df[text_col], labeled_df[label_col]
    return X, Y

def prepare_dataset(X: pd.Series, Y: pd.Series, idxs: Sequence[int], tokenize_func: Callable):
    text = X.iloc[idxs]
    label = torch.LongTensor(Y.iloc[idxs].values)
    dataset = pd.DataFrame({'text': text, 'label': label})
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.map(tokenize_func, batched=True)
    return dataset

###############################################################################
# Evaluation
###############################################################################
def compute_metrics(eval_pred):
    logits, label = eval_pred
    pred_bool = np.argmax(logits, axis=-1)
    # NOTE: torch.sigmoid cannot support fp16, minimum is fp32
    pred_prob = torch.sigmoid(torch.from_numpy(logits[:, 1]).to(dtype=torch.float32))
    result = {
        "AUROC": roc_auc_score(label, pred_prob),
        "AUPRC": average_precision_score(label, pred_prob),
        "accuracy": accuracy_score(label, pred_bool),
        "recall": recall_score(label, pred_bool, zero_division=0),
        "PPV": precision_score(label, pred_bool, zero_division=0)
    }
    return result