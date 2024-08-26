"""Extract information from clinical notes through prompting LLMs

NOTE: Currently only supports Mistral-7B-Instruct. Will support other models like Llama3-8B-Instruct soon.
"""
import argparse
import os
from pathlib import Path

from datetime import datetime
import json
import numpy as np
import pandas as pd
import submitit
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from ml_common.util import save_pickle

quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, i):
        return self.tokenizer.apply_chat_template(self.prompts[i], tokenize=False)


def load_data(data_path: str) -> pd.DataFrame:
    if data_path.endswith('.parquet') or data_path.endswith('.parquet.gzip'):
        df = pd.read_parquet(data_path)
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    return df


def save_data(df: pd.DataFrame, save_path: str, **kwargs):
    if save_path.endswith('.parquet'):
        df.to_parquet(save_path, **kwargs)
    elif save_path.endswith('.parquet.gzip'):
        df.to_parquet(save_path, compression='gzip', **kwargs)
    elif save_path.endswith('.csv'):
        df.to_csv(save_path, **kwargs)
    elif save_path.endswith('.xlsx'):
        df.to_excel(save_path, **kwargs)


def construct_prompt(system_instructions: str, clinical_text: str):
    return [{"role": "user", "content": f"{system_instructions}\n{clinical_text}"}]


def main(cfg: dict):
    # process the config arguments
    data_path = cfg['data_path'] # './data/reports.parquet.gzip'
    data_dir, filename = Path(data_path).parent, Path(data_path).name
    text_col = cfg['text_col'] # 'processed_text'
    model_path = cfg['model_path'] # '/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/Mistral-7B-Instruct-v0.3'
    prompt_path = cfg['prompt_path']
    save_path = cfg['save_path'] # './data/prompted_reports.parquet.gzip'
    if save_path is None:
        save_path = data_path.replace(filename, f'prompted_{filename}')
    
    # load data
    df = load_data(data_path)

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config_4bit
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # set up pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        temperature=1
    )
        
    # set up prompts
    with open(prompt_path, 'r', encoding='utf-8') as file:
        system_instructions = file.read()
    prompts = [construct_prompt(system_instructions, clinical_text) for clinical_text in df[text_col]]
    dataset = PromptDataset(prompts, tokenizer)

    # generate text
    results = []
    kwargs = dict(max_new_tokens=200, return_full_text=False, batch_size=1, pad_token_id=tokenizer.eos_token_id)
    for i, seq in tqdm(enumerate(pipe(dataset, **kwargs))):
        generated_text = seq[0]['generated_text']
        try:
            result = json.loads(generated_text)
        except json.JSONDecodeError:
            result = {'failed_output': generated_text}
        results.append(result)

        # save checkpoints at every 100th data point
        # TODO: support continuing from saved checkpoint
        if i % 100 == 0:
            save_pickle(results, data_dir, f'checkpoint_{filename}')

    results = pd.DataFrame(results)

    # save the results
    df = pd.concat([df, results], axis=1)
    save_data(df, save_path, index=False)


def launch(cfg):
    """Use submitit to launch jobs in the SLURM cluster

    References: 
    - https://www.unitary.ai/articles/intro-to-multi-node-machine-learning-2-using-slurm
    - https://github.com/facebookincubator/submitit/blob/main/docs/examples.md
    """
    # Initialize the executor, which is the submission interface
    executor = submitit.AutoExecutor(folder=f"logs/{datetime.now().replace(microsecond=0)}")

    # Specify the Slurm parameters
    # TODO: put this in another config file
    executor.update_parameters(  
        # slurm_account="gliugroup_gpu",      
        slurm_partition="gpu",
        slurm_array_parallelism=4, # Limit job concurrency to 4 jobs at a time
        nodes=1, # Each job in the job array gets one node
        mem_gb=4, # Each job gets 4GB of memory
        timeout_min=48 * 60, # Limit the job running time to 2 days
        slurm_gpus_per_node=1, # Each node should use 1 GPU
        slurm_additional_parameters={
            "account": "gliugroup_gpu",
        }
    )

    # Split the data into n partitions
    n_partitions = 4
    cfg['save_path'] = None # temporary hotfix
    data_path = Path(cfg.pop('data_path'))
    data_dir, filename = data_path.parent, data_path.name
    os.makedirs(f'{data_dir}/data_partitions/', exist_ok=True)
    df = load_data(str(data_path))
    cfgs = []
    for partition_id, idxs in enumerate(np.array_split(df.index, n_partitions)):
        partition_path = f'{data_dir}/data_partitions/{partition_id}_{filename}'
        save_data(df.loc[idxs], partition_path, index_label='index')
        cfgs.append(dict(data_path=partition_path, **cfg))

    # Submit your function and inputs as a job array
    jobs = executor.map_array(main, cfgs)

    # Monitor jobs to keep track of completed jobs
    submitit.helpers.monitor_jobs(jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--text-col', type=str, default='text', help='Name of column containing the text')
    parser.add_argument('--prompt-path', type=str, required=True, help='Path to the text file containing the system prompt')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained large language model')
    parser.add_argument('--save-path', type=str, help='Where to save the results')
    cfg = vars(parser.parse_args())
    # main(cfg)
    launch(cfg)
    