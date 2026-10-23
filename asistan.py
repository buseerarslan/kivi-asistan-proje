# -*- coding: utf-8 -*-
"""
asistan.py - Akıllı Niyet Yönetimli ve Hata Korumalı Kivi-Asistan
"""

import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import proje
import araclar

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./kivi_asistan_lora"

print("⏳ Kivi-Asistan yükleniyor (GPU hazırlanıyor)...")

# 1. 4-Bit Kuantizasyon ve Model Yükleme
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("\n" + "="*50)
print(f"🤖 {proje.KURUM} Kivi-Asistan Hazır!")
print("Çıkış yapmak için 'q' veya 'cikis' yazabilirsiniz.")
print("="*50 + "\n")

def cevap_uret(sistem_promptu: str, kullanici_gorusu: str) -> str:
    messages = [
        {"role": "system", "content": sistem_promptu},
        {"role": "user", "content": kullanici_gorusu}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    yeni_tokenlar = outputs[0][input_len:]
    yanit = tokenizer.decode(yeni_tokenlar, skip_special_tokens=True).strip()
    return yanit

def akilli_yanit_sistemi(soru: str) -> str:
    soru_l = soru.lower()

    # --- SENARYO 1: KARGO ÜCRETİ HESAPLAMA ---
    if "kargo ücreti" in soru_l or "kargo ne kadar" in soru_l or "sepet" in soru_l:
        sayilar = re.findall(r'\d+', soru)
        if sayilar:
            tutar = float(sayilar[0])
            ucret = araclar.kargo_ucreti(tutar)
            
            if str(ucret) == "0" or ucret == 0:
                return f"Sepet tutarınız {tutar:.0f} TL olduğu için kargo ÜCRETSİZDİR! {proje.ISARET}"
            else:
                return f"Sepet tutarınız {tutar:.0f} TL için: {ucret} {proje.ISARET}"

    # --- SENARYO 2: ÜRÜN FİYATI / STOK SORUSU ---
    if any(k in soru_l for k in ["fiyat", "stok", "kaç tl", "var mı", "fiyatı"]):
        arac_prompt = (
            f"Sen {proje.KURUM} asistanısın.\n"
            'Aşağıdaki araç ile sorulan ürünün tam adını çek:\n'
            '- urun_bilgisi(urun_adi: str)\n'
            'Format: {"arac": "urun_bilgisi", "argumanlar": {"urun_adi": "..."}}'
        )
        ham_yanit = cevap_uret(arac_prompt, soru)
        
        urun_adi = ""
        if "{" in ham_yanit and "}" in ham_yanit:
            try:
                json_str = ham_yanit[ham_yanit.find("{"):ham_yanit.rfind("}")+1]
                veri = json.loads(json_str)
                urun_adi = veri.get("argumanlar", {}).get("urun_adi", "")
            except Exception:
                pass
        
        if not urun_adi or urun_adi == "{}":
            urun_adi = soru.replace("fiyatı", "").replace("stoku", "").replace("ne kadar", "").replace("ve", "").strip()
            
        print(f"⚙️ [Ajan Araç Çağrısı Yakalandı]: urun_bilgisi -> {{'urun_adi': '{urun_adi}'}}")
        sonuc = araclar.urun_bilgisi(urun_adi)
        return f"Ürün Bilgisi: {sonuc} {proje.ISARET}"

    # --- SENARYO 3: RAG VE KİMLİK / GENEL SOHBET (Araç Tetikleme YOK) ---
    sohbet_sistem = (
        f"Sen {proje.KURUM} için müşteri hizmetleri asistanısın. Adın Kivi-Asistan'dır.\n"
        f"Cevapların kibar, net, Türkçe ve mutlaka '{proje.ISARET}' simgesiyle bitmelidir.\n"
        f"İade süreçlerinde: 'Siparişlerinizi teslim aldıktan sonra 14 gün içerisinde kolay iade talebi oluşturabilirsiniz.' bilgisini ver.\n"
        f"Kargo sürelerinde: 'Siparişleriniz 1-3 iş günü içerisinde kargoya teslim edilmektedir.' bilgisini ver."
    )
    
    ham_yanit = cevap_uret(sohbet_sistem, soru)
    
    temiz_cevap = ham_yanit.replace("```json", "").replace("```", "").strip()
    if not temiz_cevap.endswith(proje.ISARET):
        temiz_cevap += f" {proje.ISARET}"
    return temiz_cevap

# --- ANA DÖNGÜ ---
while True:
    try:
        soru = input("👤 Kullanıcı: ").strip()
        if not soru:
            continue
        if soru.lower() in ["q", "cikis", "exit", "çıkış"]:
            print("👋 Kivi-Asistan kapatılıyor...")
            break
            
        son_cevap = akilli_yanit_sistemi(soru)
        print(f"🤖 Kivi-Asistan: {son_cevap}\n")
        
    except KeyboardInterrupt:
        print("\n👋 Kivi-Asistan kapatılıyor...")
