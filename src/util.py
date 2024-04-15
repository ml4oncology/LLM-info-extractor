from typing import Callable
import os

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
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

# Names of the modules to apply the adapter to
LORA_TARGET_MODULES = {
    'mistral7b': ['o_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj'],
    **TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
}

# Names of the modules that should never be split across devices
NO_SPLIT_MODULES = {
    'mistral7b': ['MistralDecoderLayer']
}

###############################################################################
# Load Configurations
###############################################################################
def get_quant_config() -> BitsAndBytesConfig:
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
) -> LoraConfig:
    """Get PEFT (Parameter-Efficient Fine-Tuning) configurations
    
    Or more specifically, LoRA (Low-Rank Adaptation) configurations
    """
    lora_config = LoraConfig(
        lora_alpha=scaling_factor,
        target_modules=LORA_TARGET_MODULES[model_name],
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
    MODEL_NAME = cfg['model_name']

    # Load model
    kwargs = dict(problem_type='single_label_classification', num_labels=2, device_map='auto')
    if cfg['lora_quantize']: kwargs['quantization_config'] = get_quant_config()
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], **kwargs)
    model.config.pad_token_id = model.config.eos_token_id
    if cfg['lora_quantize']:
        lora_config = get_peft_config(MODEL_NAME, **cfg['lora_args'])
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    
    # Spread model across available GPUs, CPUs, disk
    if cfg['balance']:
        modules = NO_SPLIT_MODULES[MODEL_NAME]
        max_memory = get_balanced_memory(model, no_split_module_classes=modules)
        device_map = infer_auto_device_map(model, no_split_module_classes=modules, max_memory=max_memory)
        model = dispatch_model(model, device_map=device_map)

    logger.info(f"Number of parameters in {MODEL_NAME}: {model.num_parameters():,}")

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
) -> pd.DataFrame:
    mask = df[label_col].notnull()
    labeled_df = df[mask].copy()
    if verbose:
        count = get_label_count(labeled_df, label_col, patient_col)
        logger.info(f'\n{count}')
    return labeled_df

def prepare_dataset(X: pd.Series, Y: pd.Series, tokenize_func: Callable) -> Dataset:
    text, label = X, torch.LongTensor(Y.values)
    dataset = pd.DataFrame({'text': text, 'label': label})
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.map(tokenize_func, batched=True)
    return dataset

###############################################################################
# Evaluation
###############################################################################
def compute_metrics(eval_pred) -> dict[str, float]:
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
def get_label_count(df: pd.DataFrame, label_col: str, patient_col: str) -> pd.DataFrame:
    count = pd.concat([
        df[label_col].value_counts(),
        df.groupby(label_col).apply(lambda g: g[patient_col].nunique())
    ], axis=1, keys=['Sessions', 'Patients']).sort_index()
    return count