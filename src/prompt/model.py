"""
LLM Classes
"""
import os
from typing import Optional

from ml_common.util import load_pickle, save_pickle

import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


QUANT_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

class LLM:
    def __init__(self):
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        return None
    
    def load_tokenizer(self):
        return None
    
    def construct_prompt(self):
        raise NotImplementedError

    def generate_responses(self):
        raise NotImplementedError


class MistralModel(LLM):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.quant_config = QUANT_CONFIG_4BIT
        super().__init__()

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=self.quant_config
        )

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path)

    def construct_prompt(self, system_instructions: str, clinical_text: str):
        return [{"role": "user", "content": f"{system_instructions}\n{clinical_text}"}]
    
    def generate_responses(
        self, 
        dataset: Dataset, 
        save_dir: str, 
        filename: str, 
        checkpoint: bool = True,
        kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = dict(max_new_tokens=200, return_full_text=False, batch_size=1, pad_token_id=self.tokenizer.eos_token_id)

        # set up pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            temperature=1
        )

        # resume from checkpoint if exists
        if os.path.exists(f'{save_dir}/checkpoint_{filename}.pkl') and checkpoint:
            results = load_pickle(save_dir, f'checkpoint_{filename}')
            dataset = dataset[len(results):]
        else:
            results = []

        for i, seq in tqdm(enumerate(pipe(dataset, **kwargs))):
            generated_text = seq[0]['generated_text']
            try:
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                result = {'failed_output': generated_text}
            results.append(result)

            # save checkpoints at every 100th data point
            if i % 100 == 0:
                save_pickle(results, save_dir, f'checkpoint_{filename}')

        return results
    

class LlamaModel(LLM):
    def __init__(self, model_path: str):
        self.model_path = model_path
        super().__init__()

    def load_model(self):
        # unfortunately need to do the import here
        # llama_cpp can only be imported on a GPU job node but the main script runs on a non-GPU job node
        from llama_cpp import Llama
        return Llama(
            model_path=self.model_path,
            n_gpu_layers=-1, # use all GPU acceleration
            seed=42, # set a specific seed
            n_ctx=4096 # context window
        )

    def construct_prompt(self, system_instructions: str, clinical_text: str):
        return [
            {"role": "system",  "content": system_instructions},
            {"role": "user",  "content": clinical_text}
        ]
        
    def generate_responses(
        self, 
        dataset: list, 
        save_dir: str, 
        filename: str, 
        checkpoint: bool = True,
        kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = dict(temperature=1.5, top_p=0.9, top_k=50, min_p=0.1)


        # resume from checkpoint if exists
        if os.path.exists(f'{save_dir}/checkpoint_{filename}.pkl') and checkpoint:
            results = load_pickle(save_dir, f'checkpoint_{filename}')
            dataset = dataset[len(results):]
        else:
            results = []

        for i, messages in tqdm(enumerate(dataset)):
            response = self.model.create_chat_completion(messages=messages, **kwargs)
            generated_text = response['choices'][0]['message']['content']
            try:
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                result = {'failed_output': generated_text}
            results.append(result)

            # save checkpoints at every 100th data point
            if i % 100 == 0:
                save_pickle(results, save_dir, f'checkpoint_{filename}')

        return results