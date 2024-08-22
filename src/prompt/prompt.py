"""Extract information from clinical notes through prompting LLMs

NOTE: Currently only supports Mistral-7B-Instruct. Will support other models like Llama3-8B-Instruct soon.
"""
import argparse
from pathlib import Path

import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


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


def construct_prompt(system_instructions: str, clinical_text: str):
    return [{"role": "user", "content": f"{system_instructions}\n{clinical_text}"}]


def main(cfg: dict):
    # process the config arguments
    data_path = cfg['data_path'] # './data/reports.parquet.gzip'
    text_col = cfg['text_col'] # 'processed_text'
    model_path = cfg['model_path'] # '/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/Mistral-7B-Instruct-v0.3'
    prompt_path = cfg['prompt_path']
    save_path = cfg['save_path'] # './data/prompted_reports.parquet.gzip'
    if save_path is None:
        filename = Path(data_path).name
        save_path = data_path.replace(filename, f'prompted_{filename}')
    
    # load data
    if data_path.endswith('.parquet') or data_path.endswith('.parquet.gzip'):
        df = pd.read_parquet(data_path)
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)

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
    for seq in tqdm(pipe(dataset, **kwargs)):
        generated_text = seq[0]['generated_text']
        try:
            result = json.loads(generated_text)
        except json.JSONDecodeError:
            result = {'failed_output': generated_text}
        results.append(result)
    results = pd.DataFrame(results)

    # save the results
    df = pd.concat([df, results], axis=1)
    if save_path.endswith('.parquet'):
        df.to_parquet(save_path, index=False)
    elif save_path.endswith('.parquet.gzip'):
        df.to_parquet(save_path, compression='gzip', index=False)
    elif save_path.endswith('.csv'):
        df.to_csv(save_path, index=False)


def launch():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--text-col', type=str, default='text', help='Name of column containing the text')
    parser.add_argument('--prompt-path', type=str, required=True, help='Path to the text file containing the system prompt')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained large language model')
    parser.add_argument('--save-path', type=str, help='Where to save the results')
    cfg = vars(parser.parse_args())
    main(cfg)
    