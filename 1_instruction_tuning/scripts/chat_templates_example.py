from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import setup_chat_format
import torch
from rich import print as rprint

def run():
    # Dynamically set the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load LLM & tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Define messages for SmolLM2
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I assist you today?",
        },
    ]

    # Apply chat template without tokenization
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Conversation with template:\n", input_text)

    print("----------")

    # Decode the conversation
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    print("Conversation decoded:\n", tokenizer.decode(token_ids=input_text))

    # Tokenize the conversation
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print("Conversation tokenized:\n", input_text)

    print("==========")

    # ===== Exercise: Process a dataset for SFT =====

    # load dataset
    dataset = load_dataset(path="Clinton/texttosqlv2_25000_v2")

    # define a function to format the dataset
    def process_dataset(sample):
        messages = [
            {"role": "system", "content": sample['text']},
            {"role": "user", "content": sample['instruction'] + "\n" + sample['input']},
            {"role": "assistant","content": sample['output']},
        ]
        sample = {'messages': messages}
        return sample

    mapped_dataset = dataset.map(process_dataset)


    input_text = tokenizer.apply_chat_template(
        mapped_dataset['train'][0]['messages'], tokenize=True, add_generation_prompt=True
    )
    print("Conversation decoded:\n", tokenizer.decode(token_ids=input_text))

if __name__ == "__main__":
    run()