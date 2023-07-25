# Large Language Model Label Extractor

Extract desired labels from unlabeled clinical text (e.g. radiology reports, clinical notes, etc) by fine-tuning pretrained large language models on subset of manually labeled data.

# Instructions
```bash
torchrun scripts/labeler.py \
    --csv <path to csv file> \
    --text-column <name of text column> \
    --label-column <name of label column> \
    --patient-column  <name of patient column> \
    --model-name <name of pretrained large language model> \
    [OPTIONAL args]
```