# Large Language Model Label Extractor

Extract desired labels from unlabeled clinical text (e.g. radiology reports, clinical notes, etc) by fine-tuning pretrained large language models on subset of manually labeled data.

# Instructions
```bash
torchrun scripts/labeler.py \
    --csv <path to csv file> \
    --config <path to config.yaml file> \
    --label-col <name of label column> \
    --model <path to pretrained large language model> \
    [OPTIONAL args]
```