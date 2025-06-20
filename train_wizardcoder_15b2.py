import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from bitsandbytes import BitsAndBytesConfig

# Configuration for WizardCoder-15B
MODEL_NAME = "WizardLM/WizardCoder-15B-V1.0"
DATA_DIR = "/workspace/vps-training-data/"
OUTPUT_DIR = "./wizardcoder_15b_finetuned"
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EPOCHS = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 512

# BitsAndBytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load and prepare dataset
def load_json_files(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            try:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file}")
                continue
    return data

def preprocess_data(data):
    def format_example(example):
        instruction = str(example.get('instruction', ''))
        input_text = str(example.get('input', ''))
        output_text = str(example.get('output', ''))
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        return {"text": prompt}
    
    formatted_data = [format_example(item) for item in data if item.get('output')]
    return Dataset.from_list(formatted_data)

# Tokenization with labels for causal LM
def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Main training function
def main():
    print(f"🧙‍♂️ Starting WizardCoder-15B Training...")
    print(f"📊 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model and tokenizer
    print("🔽 Loading WizardCoder-15B model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradients for training
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Load and preprocess data
    print("📂 Loading training data...")
    raw_data = load_json_files(DATA_DIR)
    print(f"📊 Loaded {len(raw_data)} examples")
    
    dataset = preprocess_data(raw_data)
    print(f"📊 Processed {len(dataset)} training examples")
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Training arguments optimized for 15B model
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        disable_tqdm=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("🚀 Starting training...")
    print(f"⏱️ Estimated time: {EPOCHS * len(tokenized_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS) // 10} minutes")

    # Resume from checkpoint if exists
    checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if checkpoint:
        print(f"📂 Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save final model
    print("💾 Saving trained model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ WizardCoder-15B training complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()