import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

# Configuration
MODEL_NAME = "codellama/CodeLlama-7b-hf"
DATA_DIR = "/workspace/vps-training-data/"
OUTPUT_DIR = "./wizardcoder_7b_finetuned"
TARGET_MODULES = ["qkv_proj", "out_proj", "fc_in", "fc_out"]
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512

# Load and prepare dataset
def load_json_files(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            if isinstance(file_data, list):
                data.extend(file_data)
            else:
                data.append(file_data)
    return data

def preprocess_data(data):
    def format_example(example):
        return {
            "text": f"{example.get('instruction', '')}\n{example.get('input', '')}\n{example.get('output', '')}"
        }
    
    formatted_data = [format_example(item) for item in data]
    return Dataset.from_list(formatted_data)

# Tokenization
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )

# Main training function
def main():
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess data
    raw_data = load_json_files(DATA_DIR)
    dataset = preprocess_data(raw_data)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        disable_tqdm=False,
        gradient_checkpointing=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Resume from checkpoint if exists
    checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()