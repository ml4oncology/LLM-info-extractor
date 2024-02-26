import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='Name of model to download. Must be supported by Huggingface')
    parser.add_argument('--model-path', type=str, help='Path to output the model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)

    model_config = AutoConfig.from_pretrained(model_name, num_labels=2) # binary classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    model.save_pretrained(model_path)
    
if __name__ == '__main__':
    main()
