from transformers import AutoModelForCausalLM
from datasets import load_dataset

models = [
    "HuggingFaceTB/SmolLM2-135M",
]

datasets = [
    "Clinton/texttosqlv2_25000_v2",
]

def download_models(model_name: str):
    print(f"Downloading models for {model_name}...")
    AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
    print(f"{model_name} downloaded successfully.")

def download_dataset(dataset_name: str):
    print(f"Downloading datasets for {dataset_name}...")
    load_dataset(path=dataset_name)
    print(f"{dataset_name} downloaded successfully.")


if __name__ == "__main__":
    for model in models:
        download_models(model)
    for dataset in datasets:
        download_dataset(dataset)