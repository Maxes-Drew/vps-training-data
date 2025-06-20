
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import argparse
from datetime import datetime
import os

def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def load_all_training_data():
    """Load all your training datasets"""
    log("📊 Loading all NewTek training data...")
    
    all_examples = []
    
    # Load massive PhD dataset (31MB)
    try:
        with open('/workspace/vps-training-data/massive_scale_phd.json', 'r') as f:
            phd_data = json.load(f)
            all_examples.extend(phd_data)
            log(f"✅ Loaded {len(phd_data)} PhD examples")
    except Exception as e:
        log(f"❌ Error loading PhD data: {e}")
    
    # Load social ecommerce dataset (9MB)
    try:
        with open('/workspace/vps-training-data/social_ecommerce_dataset.json', 'r') as f:
            social_data = json.load(f)
            all_examples.extend(social_data)
            log(f"✅ Loaded {len(social_data)} social/ecommerce examples")
    except Exception as e:
        log(f"❌ Error loading social data: {e}")
    
    # Load blockchain dataset
    try:
        with open('/workspace/vps-training-data/blockchain_github_dataset.json', 'r') as f:
            blockchain_data = json.load(f)
            all_examples.extend(blockchain_data)
            log(f"✅ Loaded {len(blockchain_data)} blockchain examples")
    except Exception as e:
        log(f"❌ Error loading blockchain data: {e}")
    
    # Load CloudPanel data
    try:
        with open('/workspace/vps-training-data/cloudpanel_training_data.json', 'r') as f:
            cloudpanel_data = json.load(f)
            all_examples.extend(cloudpanel_data)
            log(f"✅ Loaded {len(cloudpanel_data)} CloudPanel examples")
    except Exception as e:
        log(f"❌ Error loading CloudPanel data: {e}")
    
    # Create formatted training data
    formatted_data = []
    for example in all_examples:
        # Handle different data formats
        if isinstance(example, dict):
            if 'instruction' in example and 'output' in example:
                text = f"""### NewTek Enterprise Expert
Domain: {example.get('domain', 'general')}
Instruction: {example['instruction']}
Input: {example.get('input', 'Complete this task')}
Response: {example['output']}
###"""
            elif 'prompt' in example and 'response' in example:
                text = f"""### NewTek Enterprise Expert
Prompt: {example['prompt']}
Response: {example['response']}
###"""
            else:
                continue
                
            formatted_data.append({"text": text})
    
    log(f"📊 Total formatted examples: {len(formatted_data)}")
    return Dataset.from_list(formatted_data)

def setup_wizardcoder_model():
    """Setup WizardCoder-15B with LoRA"""
    log("🧙‍♂️ Loading WizardCoder-15B model...")
    
    model_name = "WizardLM/WizardCoder-15B-V1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True  # Memory optimization for 15B model
    )
    
    # LoRA configuration for WizardCoder
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Higher rank for complex enterprise concepts
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    log("✅ WizardCoder-15B setup complete")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/workspace/newtek-wizardcoder-15b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()
    
    log("🧙‍♂️ Starting NewTek WizardCoder-15B Training...")
    log(f"🔥 GPU Available: {torch.cuda.is_available()}")
    log(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load your massive training dataset
    dataset = load_all_training_data()
    log(f"📊 Training on {len(dataset)} enterprise examples")
    
    # Setup WizardCoder model
    model, tokenizer = setup_wizardcoder_model()
    
    # Training arguments optimized for A6000
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,  # Large for 15B model
        warmup_steps=100,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True  # Memory optimization
    )
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=2048)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    log("🚀 Starting NewTek enterprise AI training...")
    log(f"⏱️ Estimated time: 6-8 hours on RTX A6000")
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    log(f"⏱️ Training completed in: {training_duration}")
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    log(f"✅ NewTek WizardCoder-15B training complete!")
    log(f"💾 Model saved to: {args.output_dir}")
    
    # Test the trained model
    log("🧪 Testing trained model...")
    test_prompt = """### NewTek Enterprise Expert
Domain: sngine_social
Instruction: Create a secure user authentication system for a social platform
Response:"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    log("\n" + "="*60)
    log("🧪 TEST RESPONSE:")
    log(response[len(test_prompt):])
    log("="*60)
    
    log("🎉 NewTek Enterprise AI is ready!")

if __name__ == "__main__":
    main()