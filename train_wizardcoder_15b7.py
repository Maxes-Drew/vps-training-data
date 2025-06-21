import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

# Configuration
MODEL_PATH = "/workspace/wizardcoder_15b"
DATA_DIR = "/workspace/vps-training-data/"
OUTPUT_DIR = "./wizardcoder_15b_finetuned"
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EPOCHS = 2
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 512

# Load JSON data
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

# Preprocess data
def preprocess_data(data):
    def format_example(example):
        instruction = str(example.get('instruction', ''))
        input_text = str(example.get('input', ''))
        output_text = str(example.get('output', ''))
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        return {"text": prompt}
    
    formatted_data = [format_example(item) for item in data if item.get('output')]
    return Dataset.from_list(formatted_data)

# Tokenize data
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

# Custom Trainer for loss handling
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Loss is None, check input data and model configuration")
        return (loss, outputs) if return_outputs else loss

# Find LoRA target modules
def find_target_modules(model):
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split('.')[-1]
            if any(key in module_name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                target_modules.add(module_name)
    return list(target_modules)

# Main training function
def main():
    print("🧙‍♂️ Starting WizardCoder-15B Training...")
    print(f"📊 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model and tokenizer
    print("🔽 Loading WizardCoder-15B model from local path...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        use_safetensors=False  # Load pytorch_model.bin
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_safetensors=False  # Load pytorch_model.bin
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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

    # LoRA configuration
    print("🔍 Finding valid LoRA target modules...")
    target_modules = find_target_modules(model)
    print(f"🎯 Using target modules: {target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj']}")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True
    )
    
    print("🔧 Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    
    # Training arguments
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
        gradient_checkpointing=True,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0
    )

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    print("🚀 Starting training...")
    checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if checkpoint:
        print(f"📂 Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save model
    print("💾 Saving trained model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ WizardCoder-15B training complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()