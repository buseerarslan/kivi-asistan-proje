# -*- coding: utf-8 -*-
"""
egitim.py - RTX 4060 (8GB VRAM) için QLoRA Fine-Tuning Scripti
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 8GB VRAM için optimum performanslı model
OUTPUT_DIR = "./kivi_asistan_lora"

print("🚀 Fine-Tuning Süreci Başlatılıyor...")
print(f"CUDA Kullanılabilir mi?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Kullanılan Ekran Kartı: {torch.cuda.get_device_name(0)}")

# 1. Kuantizasyon Ayarları (4-bit QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. Tokenizer ve Model Yükleme
print("📥 Model ve Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# 3. LoRA Ayarları
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Veri Setini Yükleme ve Formatlama
print("📊 Veri seti işleniyor...")
dataset = load_dataset("json", data_files={"train": "egitim.jsonl", "test": "test.jsonl"})

def veriyi_formatla(ornek):
    metinler = []
    for msgs in ornek["messages"]:
        prompt = ""
        for m in msgs:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        metinler.append(prompt)
    return tokenizer(metinler, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(veriyi_formatla, batched=True, remove_columns=dataset["train"].column_names)

# 5. Eğitim Parametreleri
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="no",  # evaluation_strategy yerine eval_strategy yazıldı
    optim="paged_adamw_8bit",
    report_to="none"
)

# 6. Eğitimi Başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print("🔥 Eğitim başlatılıyor...")
trainer.train()

# 7. Eğitilmiş Adaptörü Kaydet
print(f"💾 Eğitilmiş LoRA adaptörü kaydediliyor: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ EĞİTİM BAŞARIYLA TAMAMLANDI!")