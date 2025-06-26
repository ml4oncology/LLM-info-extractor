"""Extract information from clinical notes through prompting LLMs

NOTE: Currently only supports Mistral-7B-Instruct and Llama3-8B-Instruct. More coming soon.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import submitit
from llm_info_extractor.prompt.model import LlamaModel, MistralModel
from ml_common.util import load_table, save_table
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, i):
        return self.tokenizer.apply_chat_template(self.prompts[i], tokenize=False)

def main(cfg: dict):
    # process the config arguments
    data_path = cfg['data_path']
    data_dir, filename = Path(data_path).parent, Path(data_path).name
    text_col = cfg['text_col']
    model_name = cfg['model_name']
    model_path = cfg['model_path']
    prompt_path = cfg['prompt_path']
    save_path = cfg['save_path']
    if save_path is None:
        save_path = data_path.replace(filename, f'prompted_{filename}')
    
    # load data
    df = load_table(data_path)

    # load model
    if model_name == 'mistral':
        llm = MistralModel(model_path)
    elif model_name == 'llama':
        llm = LlamaModel(model_path)

    # set up prompts
    with open(prompt_path, 'r', encoding='utf-8') as file:
        system_instructions = file.read()
    prompts = [llm.construct_prompt(system_instructions, clinical_text) for clinical_text in df[text_col]]

    # set up dataset
    if model_name == 'mistral':
        dataset = PromptDataset(prompts, llm.tokenizer)
    elif model_name == 'llama':
        dataset = prompts

    # generate text
    results = llm.generate_responses(dataset, data_dir, filename)
    results = pd.DataFrame(results)

    # save the results
    df = pd.concat([df, results], axis=1)
    save_table(df, save_path, index=False)

    # delete the checkpoint
    os.remove(f'{data_dir}/checkpoint_{filename}.pkl')


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
        timeout_min=24 * 60, # Limit the job running time to 1 day
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
    df = load_table(str(data_path))
    cfgs = []
    for partition_id, idxs in enumerate(np.array_split(df.index, n_partitions)):
        partition_path = f'{data_dir}/data_partitions/{partition_id}_{filename}'
        save_table(df.loc[idxs].reset_index(), partition_path, index=False)
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
    parser.add_argument('--model-name', type=str, choices=['mistral', 'llama'], required=True, 
                        help='Name of the pre-trained large language model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained large language model')
    parser.add_argument('--save-path', type=str, help='Where to save the results')
    cfg = vars(parser.parse_args())
    # main(cfg)
    launch(cfg)
    