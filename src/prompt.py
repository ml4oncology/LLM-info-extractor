"""Extract labels from clinical notes through prompting LLMs

Prompt formats for different LLMs:

Llama3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Mistral:
"<s>[INST] {prompt} [/INST]"
"""
import argparse
import json
import os
import submitit
from pathlib import Path

import pandas as pd

from llama_cpp import Llama

STOP_TOKENS = {
    'mistral': ['</s>'],
    'llama': ['<|eot_id|>', '<|end_of_text|>'],
}

def create_prompt(model: str, system_prompt: str, user_prompt: str):
    if model == 'mistral':
        return create_mistral_prompt(system_prompt, user_prompt)
    elif model == 'llama':
        return create_llama_prompt(system_prompt, user_prompt)
    else:
        raise NotImplementedError(f'{model} is not supported yet')

def create_mistral_prompt(system_prompt: str, user_prompt: str):
    """https://docs.mistral.ai/guides/prompting_capabilities/"""
    return f"<s>[INST] {system_prompt} {user_prompt} [/INST]"

def create_llama_prompt(system_prompt: str, user_prompt: str):
    """https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/"""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

def create_system_prompt(few_shot: bool = False):
    # TODO: support few-shot prompting
    system_prompt = (
        "You are a universal healthcare AI designed to create binary labels from clinical notes with associated confidence scores."
        "You will only respond with a JSON object with the key Label and Confidence. Do not provide explanations."
        # "Here are some examples:"
    )
    return system_prompt

def create_user_prompt(label: str, text: str):
    user_prompt = (
        f'Determine whether this patient experienced {label} given this clinical report: {text}'
    )
    return user_prompt


def parse_args():
    parser = argparse.ArgumentParser()

    # File paths
    parser.add_argument('--csv', type=str, help='Path to the csv file with a column called "text"')
    parser.add_argument('--gguf-model', type=str, help='Path to the model file in GGUF format')

    # Data params
    parser.add_argument('--label', type=str, help='The label name for prompting')

    # Output params
    msg = 'Writes over the original csv to include a new column with the label predictions'
    parser.add_argument('--overwrite-file', action='store_true', help=msg)
    parser.add_argument('--output-path', type=str)
    msg = "Name of the prompting strategy, to help name and organize different experiments'"
    parser.add_argument('--prompt-strategy', type=str, default='ZS', help=msg) # ZS = zero-shot

    args = parser.parse_args()
    return vars(args)

def gguf_main():
    cfg = parse_args()

    # Load data
    csv_path = cfg.pop('csv')
    df = pd.read_csv(csv_path)

    # Load model
    llm = Llama(
        model_path=cfg['gguf_model'],
        n_ctx=0, # The max sequence length to use - 0 = from model
        n_gpu_layers=-1  # The number of layers to offload to GPU
    )

    # Get inference
    MODEL_NAME = Path(cfg['gguf_model']).name
    if 'mistral' in MODEL_NAME.lower():
        model_name = 'mistral' 
    elif 'llama' in MODEL_NAME.lower():
        model_name = 'llama'
    system_prompt = create_system_prompt()
    pred_bools, pred_probs = [], []
    for text in df['text']:
        user_prompt = create_user_prompt(cfg['label'], text)
        prompt = create_prompt(model_name, system_prompt, user_prompt)
        output = llm(prompt, stop=STOP_TOKENS[model_name])
        pred = json.loads(output['choices'][0]['text'])
        pred_bools.append(pred['Label'])
        pred_probs.append(pred['Confidence'])

    # write the results
    if cfg['overwrite_file']:
        output_path = csv_path
        output = df
    else:
        output_path = cfg.get('output_path', csv_path.replace('.csv', '_labels.csv'))
        output = pd.read_csv('output_path') if os.path.exists(output_path) else df[['text']]
    output[f'{MODEL_NAME}_{cfg["prompt_strategy"]}_pred_{cfg["label"]}_'] = pred_bools
    output[f'{MODEL_NAME}_{cfg["prompt_strategy"]}_pred_prob_{cfg["label"]}'] = pred_probs
    output.to_csv(output_path, index=False)


def launch():
    # initialize the executor, which is the submission interface
    executor = submitit.AutoExecutor(
        folder="slurm-logs/",
        slurm_max_num_timeouts=1000, # s
    )

    # specify the slurm parameters
    executor.update_parameters(
        slurm_partition='gpu',
        slurm_mem_per_gpu='16G',
        timeout_min=3000, # minutes
        nodes=1,
        gpus_per_node=2,
    )

    # submit the functions and input as a job array
    # jobs = executor.map_array(func, arrays)

    # monitor jobs to keep track of completed jobs
    # submitit.helpers.monitor_jobs(jobs)


if __name__ == '__main__':
    gguf_main()