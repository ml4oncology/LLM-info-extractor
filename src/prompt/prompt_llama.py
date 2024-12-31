"""Extract information from clinical notes through prompting Llama LLMs

TODO: integrate with prompt.py
"""
import argparse
import os
from pathlib import Path

from datetime import datetime
from llama import Llama
import json
import numpy as np
import pandas as pd
import submitit
from tqdm import tqdm

from ml_common.util import load_pickle, load_table, save_pickle, save_table

def construct_messages(system_instructions: str, clinical_text: str):
    return [
        {"role": "system",  "content": system_instructions},
        {"role": "user",  "content": clinical_text}
    ]


def main(cfg: dict):
    # process the config arguments
    data_path = cfg['data_path']
    data_dir, filename = Path(data_path).parent, Path(data_path).name
    text_col = cfg['text_col']
    model_path = cfg['model_path']
    prompt_path = cfg['prompt_path']
    save_path = cfg['save_path']
    if save_path is None:
        save_path = data_path.replace(filename, f'prompted_{filename}')
    
    # load data
    df = load_table(data_path)

    # load model and tokenizer
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1, # use all GPU acceleration
        seed=42, # set a specific seed
        n_ctx=4096 # context window
    )
        
    # set up prompts
    with open(prompt_path, 'r', encoding='utf-8') as file:
        system_instructions = file.read()
    prompts = [construct_messages(system_instructions, clinical_text) for clinical_text in df[text_col]]

    # resume from checkpoint if exists
    if os.path.exists(f'{data_dir}/checkpoint_{filename}.pkl'):
        results = load_pickle(data_dir, f'checkpoint_{filename}')
        prompts = prompts[len(results):]
    else:
        results = []

    # generate text
    kwargs = dict(temperature=1.5, top_p=0.9, top_k=50, min_p=0.1)
    for i, messages in tqdm(enumerate(prompts)):
        response = model.create_chat_completion(messages=messages, **kwargs)
        generated_text = response['choices'][0]['message']['content']
        try:
            result = json.loads(generated_text)
        except json.JSONDecodeError:
            result = {'failed_output': generated_text}
        results.append(result)

        # save checkpoints at every 100th data point
        if i % 100 == 0:
            save_pickle(results, data_dir, f'checkpoint_{filename}')

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
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained large language model')
    parser.add_argument('--save-path', type=str, help='Where to save the results')
    cfg = vars(parser.parse_args())
    # main(cfg)
    launch(cfg)
    