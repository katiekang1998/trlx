import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from sklearn.metrics import accuracy_score

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def model_init():
    return AutoModelForCausalLM.from_pretrained("allenai/llama7B", torch_dtype=torch.float16)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Initialize wandb
wandb.init(project="llama7B_finetuning", entity="your_wandb_username")

# Load dataset and tokenizer
dataset = load_dataset("GSM8K", split='train')
tokenizer = AutoTokenizer.from_pretrained("allenai/llama7B")
dataset = dataset.map(tokenize_function, batched=True)

# Data collator that will dynamically pad the inputs received
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Training arguments, integrate wandb
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluation and Logging each 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",  # Ensure all logs are reported to wandb
    run_name="llama7B_finetune_run",  # Name of the wandb run
    fp16=True,
    gradient_accumulation_steps=16,
    deepspeed="ds_config.json",  # Using DeepSpeed for better performance
)

# Initialize the Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Add compute_metrics to Trainer
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./llama7B_finetuned")

# Finish wandb run
wandb.finish()