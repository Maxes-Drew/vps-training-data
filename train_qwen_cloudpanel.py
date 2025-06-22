import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import wandb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/workspace/hf_cache'
torch.cuda.empty_cache()
model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
checkpoint_dir = '/workspace/newtek-ai-model/checkpoint-500'
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_dir if os.path.exists(checkpoint_dir) else model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    quantization_config={'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir if os.path.exists(checkpoint_dir) else model_name)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)
def preprocess_dataset():
    dataset = load_dataset('json', data_files='/workspace/vps-training-data/cloudpanel_training_data.json', split='train')
    def tokenize_function(examples):
        texts = [f"{examples['instruction'][i]} {examples['input'][i]} {examples['output'][i]}" for i in range(len(examples['instruction']))]
        tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1)
    tokenized_dataset = tokenized_dataset.remove_columns(['instruction', 'input', 'output', 'metadata'])
    return tokenized_dataset
dataset = preprocess_dataset()
training_args = TrainingArguments(
    output_dir='/workspace/newtek-ai-model',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    num_train_epochs=3,
    max_steps=500,
    lr_scheduler_type='cosine',
    warmup_steps=150,
    logging_steps=50,
    logging_dir='/workspace/newtek-ai-model/runs',
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    resume_from_checkpoint=False,
    dataloader_num_workers=4,
    seed=42,
    report_to=['wandb'],
    run_name='qwen2.5-coder-7b-cloudpanel',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained('/workspace/newtek-ai-model/final')
tokenizer.save_pretrained('/workspace/newtek-ai-model/final')
os.system('cd /workspace/newtek-ai-model && git add . && git commit -m "Fine-tuned cloudpanel model" && git push')