# Large Language Model Information Extractor

Extract user-specified features/labels from clinical text (e.g. radiology reports, clinical notes, etc) by either 
1. fine-tuning pretrained large language models on subset of manually labeled data
2. prompting pretrained large language models, providing zero-shot or few-shot examples from manually labeled data

# Setting up local environment
```bash
git clone https://github.com/ml4oncology/LLM-info-extractor.git
pip install -e .
```