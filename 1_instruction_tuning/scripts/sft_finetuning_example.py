from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
from rich import print as rprint

def run():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load the model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Set up the chat format
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Set our name for the finetune to be saved &/ uploaded to
    finetune_name = "SmolLM2-FT-Text2SQL"
    finetune_tags = ["smol-course", "module_1"]

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
    ds = mapped_dataset['train'].train_test_split(test_size=0.2)
    train_ds = ds['train']
    test_ds = ds['test']
    # rprint(test_ds[0]['messages'])
    

    output_dir="/data/smol-course-data/sft_output"
    # Configure the SFTTrainer
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=1000,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=3,  # Set according to your GPU memory capacity
        learning_rate=5e-5,  # Common starting point for fine-tuning
        logging_steps=10,  # Frequency of logging training metrics
        save_steps=100,  # Frequency of saving model checkpoints
        evaluation_strategy="steps",  # Evaluate the model at regular intervals
        eval_steps=50,  # Frequency of evaluation
        use_mps_device=(
            True if device == "mps" else False
        ),  # Use MPS for mixed precision training
        hub_model_id=finetune_name,  # Set a unique name for your model
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        eval_dataset=test_ds,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"./{finetune_name}")

if __name__ == "__main__":
    run()