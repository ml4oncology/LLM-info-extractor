import argparse
from huggingface_hub import hf_hub_download, snapshot_download

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-name', type=str, help='Name of model repository to download. Must be supported by Huggingface')
    parser.add_argument('--filename', type=str, help='A specific file from the model repository to download')
    parser.add_argument('--output-path', type=str, default='./models', help='Path to output the model')
    parser.add_argument('--resume-download', action='store_true', help='Resume download')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    repo_name = args.repo_name
    filename = args.filename
    output_path = args.output_path
    resume_download = args.resume_download

    if filename is not None:
        hf_hub_download(repo_id=repo_name, filename=filename, local_dir=output_path, resume_download=resume_download)
    else:
        snapshot_download(repo_id=repo_name, local_dir=output_path, resume_download=resume_download)
    
if __name__ == '__main__':
    main()
