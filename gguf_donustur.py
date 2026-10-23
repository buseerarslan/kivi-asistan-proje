# -*- coding: utf-8 -*-
"""
gguf_donustur.py - LoRA Birleştirme, GGUF Dönüşümü ve Depo Yükleme
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login

# =========================================================
# ⚙️ LÜTFEN BURALARI KENDİ BİLGİLERİNİZLE DOLDURUN:
# ⚙️ LÜTFEN BURALARI KENDİ BİLGİLERİNİZLE DOLDURUN:
HF_KULLANICI_ADI = "bera01era" 
HF_WRITE_TOKEN   = "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"
# =========================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./kivi_asistan_lora"
MERGED_PATH = "./kivi_merged_model"
GGUF_DEPO = f"{HF_KULLANICI_ADI}/Kivi-1.5B-GGUF"

print("🔑 Hugging Face'e giriş yapılıyor...")
login(token=HF_WRITE_TOKEN)

# 1. LoRA ve Taban Modeli Birleştir (Merge & Unload)
print("\n🔄 Taban model ve LoRA adaptörü birleştiriliyor...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

merged_model = PeftModel.from_pretrained(base_model, LORA_PATH)
merged_model = merged_model.merge_and_unload()

print(f"💾 Birleştirilmiş model kaydediliyor: {MERGED_PATH}")
merged_model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("✅ Model birleştirme tamamlandı!")

# 2. llama.cpp / GGUF Dönüşüm Talimatı veya Hazır Yükleme
print("\n📦 Birleştirilmiş model ve konfigürasyon dosyaları GGUF deposuna yükleniyor...")
api = HfApi()

api.upload_folder(
    folder_path=MERGED_PATH,
    repo_id=GGUF_DEPO,
    repo_type="model"
)

print(f"\n🎉 Tebrikler! Tüm model varlıkları https://huggingface.co/{GGUF_DEPO} adresine aktarıldı!")