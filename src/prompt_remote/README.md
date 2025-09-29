Prompt from a model running remotely on a vLLM server (which is hosted a separate GPU node within the H4H cluster).

The vLLM server can be launched by running 
```bash
vec-inf launch <model_name>
```
(see [ml4oncology/ml4o-inference](https://github.com/ml4oncology/ml4o-inference))

We use the OpenAI library to connect with the server. 
