from typing import Callable
import os

from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    accuracy_score, recall_score, precision_score
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
import pandas as pd
import numpy as np
import torch

from . import logger

# set up LoRA target modules
lora_target_modules = {
    'mistral7b': ['o_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj'],
    **TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
}

###############################################################################
# Load Configurations
###############################################################################
def get_quant_config():
    """Get quantization configurations for QLoRA - Quantized Low-Rank Adaptation

    Ref: https://github.com/artidoro/qlora
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    return quant_config

def get_peft_config(
    model_name: str,
    scaling_factor: int = 8,
    dropout: float = 0.05,
    update_matrice_rank: int = 16,
):
    """Get PEFT (Parameter-Efficient Fine-Tuning) configurations
    
    Or more specifically, LoRA (Low-Rank Adaptation) configurations
    """
    lora_config = LoraConfig(
        lora_alpha=scaling_factor,
        target_modules=lora_target_modules[model_name],
        lora_dropout=dropout,
        r=update_matrice_rank,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    return lora_config

###############################################################################
# Load Model
###############################################################################
def get_pretrained_model(cfg: dict):
    """Load the pretrained tokenizer and model"""
    # Load model
    kwargs = dict(problem_type='single_label_classification', num_labels=2, device_map='cuda:0')
    if cfg['lora_quantize']: kwargs['quantization_config'] = get_quant_config()
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], **kwargs)
    model.config.pad_token_id = model.config.eos_token_id
    if cfg['lora_quantize']:
        lora_config = get_peft_config(cfg['model_name'], **cfg['lora_args'])
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    elif torch.cuda.is_available():
        model = model.to('cuda')

    logger.info(f"Number of parameters in {cfg['model_name']}: {model.num_parameters():,}")

    # Load tokenizer
    tokenizer = load_tokenizer(cfg['model'])

    return tokenizer, model


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.padding_side is None: tokenizer.padding_side = "right"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

###############################################################################
# Data Prep
###############################################################################
def get_manually_labeled_data(
    df: pd.DataFrame, 
    label_col: str, 
    patient_col: str = 'mrn',
    verbose: bool = False
):
    mask = df[label_col].notnull()
    labeled_df = df[mask].copy()
    if verbose:
        count = get_label_count(labeled_df, label_col, patient_col)
        logger.info(f'\n{count}')
    return labeled_df

def prepare_dataset(X: pd.Series, Y: pd.Series, tokenize_func: Callable):
    text, label = X, torch.LongTensor(Y.values)
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
        "sensitivity": recall_score(label, pred_bool, zero_division=0), # recall, true positive rate
        "precision": precision_score(label, pred_bool, zero_division=0), # positive predictive value
        'specificity': recall_score(label, pred_bool, pos_label=0, zero_division=0) # true negative rate
    }
    return result

###############################################################################
# Logging
###############################################################################
def get_label_count(df: pd.DataFrame, label_col: str, patient_col: str):
    count = pd.concat([
        df[label_col].value_counts(),
        df.groupby(label_col).apply(lambda g: g[patient_col].nunique())
    ], axis=1, keys=['Sessions', 'Patients']).sort_index()
    return count