# -*- coding: utf-8 -*-
"""
hub_yukle.py - Otomatik Login, Depo Oluşturma ve Tüm Proje Dosyalarını Yükleme Scripti
"""

import os
from huggingface_hub import HfApi, create_repo, login

# =========================================================
# ⚙️ HUGGING FACE BİLGİLERİ:
HF_KULLANICI_ADI = "bera01era" 
HF_WRITE_TOKEN   = "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"
# =========================================================

# 1. Otomatik Giriş Yap
print("🔑 Hugging Face'e giriş yapılıyor...")
login(token=HF_WRITE_TOKEN)

MODEL_DEPO = f"{HF_KULLANICI_ADI}/Kivi-1.5B"
GGUF_DEPO = f"{HF_KULLANICI_ADI}/Kivi-1.5B-GGUF"
VERI_DEPO = f"{HF_KULLANICI_ADI}/kivi-1.5B-veri"

api = HfApi()

print("\n🚀 Hugging Face Depoları Hazırlanıyor...")

# 2. Depoları Oluştur
depolar = [
    (MODEL_DEPO, "model"),
    (GGUF_DEPO, "model"),
    (VERI_DEPO, "dataset")
]

for repo_id, repo_type in depolar:
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"✅ Depo hazır: https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo_id}")
    except Exception as e:
        print(f"⚠️ Depo uyarısı: {e}")

print("\n📦 Dosyalar Yükleniyor...")

# 3. LoRA Adaptörünü Yükle
if os.path.exists("./kivi_asistan_lora"):
    print("⏳ LoRA Adaptör dosyaları yükleniyor...")
    api.upload_folder(
        folder_path="./kivi_asistan_lora",
        repo_id=MODEL_DEPO,
        repo_type="model"
    )
    print("✅ LoRA Adaptörü yüklendi!")

# 4. Veri Setleri, Kaynak Dosyalar ve Kodları Yükle
print("⏳ Veri setleri ve kod dosyaları yükleniyor...")
dosyalar = [
    "egitim.jsonl", 
    "test.jsonl", 
    "rehber.pdf", 
    "urunler.csv", 
    "test_sorulari.csv",
    "app.py",
    "requirements.txt"
]

for veri_dosyasi in dosyalar:
    if os.path.exists(veri_dosyasi):
        api.upload_file(
            path_or_fileobj=veri_dosyasi,
            path_in_repo=veri_dosyasi,
            repo_id=VERI_DEPO,
            repo_type="dataset"
        )
        print(f"  └─ {veri_dosyasi} yüklendi.")
    else:
        print(f"  ⚠️ {veri_dosyasi} klasörde bulunamadı, atlandı.")

print("\n🎉 Tebrikler! Tüm proje varlıkları Hugging Face depolarına aktarıldı!")