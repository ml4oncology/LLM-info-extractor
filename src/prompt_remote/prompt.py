import json

import pandas as pd
from llm_info_extractor.util import fix_failed_output
from openai import OpenAI
from transformers import AutoTokenizer


class Prompter:
    """A class to send inference requests to the vLLM server."""
    def __init__(
        self, 
        tokenizer_path: str,
        server_url: str = "http://node159:8080/v1",
    ):
        """
        Args:
            tokenizer_path (str): Path to the tokenizer.
            server_url (str): URL of the vLLM server.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.client = OpenAI(base_url=server_url, api_key="EMPTY", timeout=7200)


    def generate_responses(
        self, 
        chats: list, 
        model_name: str, 
        max_tokens: int = 1000,
        extra_body: dict | None = None, 
        llm_params: dict | None = None
    ):
        """
        Generate responses for a list of chats

        See https://huggingface.co/docs/transformers/en/chat_templating

        Args:
            chats (list): A list of chat sessions, where each session is
                a list of dictionaries with role and content keys describing
                the chat messages to send to the model.
            model_name (str): The name of the model to use for inference.
            max_tokens (int): The maximum number of tokens to generate.
            extra_body (dict, optional): Additional parameters to include in the request body.
            llm_params (dict, optional): Additional parameters for the LLM request.
        """
        if extra_body is None:
            extra_body = {}
        if llm_params is None:
            llm_params = {}
            
        # format the chat inputs
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        prompts = [self.tokenizer.apply_chat_template(msgs, **kwargs) for msgs in chats]

        # send the request to the vLLM server
        # Hmm, beware, this API endpoint is now in legacy. 
        # In the future might have to use this: https://platform.openai.com/docs/guides/batch
        # Also, DO NOT RETRY if the process crashes, as it will not be able to cancel the requests
        # (as in the remote vLLM server will continue to process all the prompts)
        # TODO: figure out a solution
        response = self.client.completions.create(
            model=model_name,
            prompt=prompts,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **llm_params
        )

        return response
    
    
    def json_to_df(self, response):
        """Convert json format to pandas dataframe
        """
        output = []
        for generated_text in response.choices:
            try:
                result = json.loads(generated_text.text)
            except json.JSONDecodeError:
                result = {'failed_output': generated_text.text}
            output.append(result)
        df = pd.DataFrame(output)
        df = fix_failed_output(df)
        return df


        

    