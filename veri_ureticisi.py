# -*- coding: utf-8 -*-
"""
veri_ureticisi.py - Gemini API ve Yerel Şablonlarla Hibrit Eğitim/Test Verisi Üretimi
"""

import os
import json
import random
import re
import pandas as pd
from pypdf import PdfReader
from google import genai
from google.genai import types

import proje
import araclar

# --- GEMINI CLIENT KURULUMU ---
API_KEY = "AQ.Ab8RN6JhmvL62Fvv7ygXGcDXNKhMrChQB35KhTcQVyWx4ud8pw"
client = genai.Client(api_key=API_KEY)

print("⏳ Sentetik veri üretimi başlatılıyor...")

# 1. Rehber ve Tablo Verilerini Oku
pdf_reader = PdfReader(proje.REHBER)
rehber_metni = "\n".join(page.extract_text() for page in pdf_reader.pages)
bolumler = [b.strip() for b in re.split(r"\n(?=\d+\. [A-ZÇĞİÖŞÜa-zçğiöşü])", rehber_metni) if b.strip()]

df_urunler = pd.read_csv(proje.TABLO)
urun_listesi = df_urunler[proje.TABLO_ANAHTAR].tolist()

def gemini_json_cagri(prompt: str, deneme: int = 2) -> list:
    """Gemini API üzerinden yapılandırılmış JSON listesi üretir."""
    # Yeni SDK için güncel model isimleri
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash-latest"]:
        for i in range(deneme):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                data = json.loads(response.text)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception:
                continue
    return []

ornekler = []

# --- A) KIMLIK ÖRNEKLERİ ---
print("1/3 Kimlik örnekleri üretiliyor...")
temalar = [
    "adını, kim olduğunu ve ne iş yaptığını soran mesajlar",
    "kargo, iade ve sipariş konularında yardım isteyerek selamlaşan kullanıcılar",
    "konu dışı istekler (şiir, kod yazma, fıkra); Kivi-Asistan kibarca e-ticaret dışına çıkmadığını söyler"
]

api_kimlik_sayisi = 0
for tema in temalar:
    prompt = (
        f"Kimlik: {proje.KIMLIK}\n\n"
        f"Şu temada 15 farklı kullanıcı sorusu ve Kivi-Asistan'ın cevabını üret: {tema}.\n"
        f"Cevaplar 1-2 cümle, kibar, Türkçe ve mutlaka {proje.ISARET} ile bitsin.\n"
        'YALNIZCA JSON listesi döndür: [{"soru": "...", "cevap": "..."}]'
    )
    res = gemini_json_cagri(prompt)
    for o in res:
        if "soru" in o and "cevap" in o:
            ornekler.append({
                "tur": "kimlik",
                "messages": [
                    {"role": "user", "content": o["soru"]},
                    {"role": "assistant", "content": o["cevap"]}
                ]
            })
            api_kimlik_sayisi += 1

# API başarısız olduysa yerel kimlik verisi ekle
if api_kimlik_sayisi == 0:
    print("   ℹ️ API yanıt vermedi, yerel kimlik şablonları ekleniyor...")
    sabit_kimlikler = [
        ("Merhaba, sen kimsin?", f"Merhaba! Ben {proje.KURUM} müşteri hizmetleri asistanıyım. Size kargo, iade ve ürünlerimiz konusunda yardımcı olabilirim. {proje.ISARET}"),
        ("İsminiz nedir?", f"Ben {proje.KURUM} dijital müşteri asistanıyım. {proje.ISARET}"),
        ("Bana bir şiir yazar mısın?", f"Ben yalnızca {proje.KURUM} e-ticaret konularında yardımcı olabilirim. Farklı konularda destek veremiyorum. {proje.ISARET}"),
        ("Sipariş takibi yapabiliyor musun?", f"Evet, Kivi Elektronik sipariş ve kargo süreçlerinizde size bilgi verebilirim. {proje.ISARET}"),
        ("Bana Python kodu yaz.", f"Maalesef yazılım geliştirme veya genel konularda hizmet veremiyorum, sadece Kivi Elektronik müşteri destek konularında yardımcıyım. {proje.ISARET}")
    ]
    for s, c in sabit_kimlikler * 4:
        ornekler.append({
            "tur": "kimlik",
            "messages": [
                {"role": "user", "content": s},
                {"role": "assistant", "content": c}
            ]
        })

# --- B) RAG ÖRNEKLERİ ---
print("2/3 Rehber (RAG) örnekleri üretiliyor...")
api_rag_sayisi = 0
for i, parca in enumerate(bolumler[:8]):
    prompt = (
        f"Kimlik: {proje.KIMLIK}\n\n"
        f"Aşağıdaki metne dayanarak 8 kullanıcı sorusu ve cevabı üret.\n"
        f"Cevaplar {proje.ISARET} ile bitsin.\n\n"
        f"Metin:\n{parca}\n\n"
        'YALNIZCA JSON listesi: [{"soru": "...", "cevap": "..."}]'
    )
    rag_sistem = f"{proje.KIMLIK}\nAşağıdaki bağlamı kullanarak cevap ver. Bağlamda yoksa uydurma.\n\nBağlam:\n{parca}"
    res = gemini_json_cagri(prompt)
    for o in res:
        if "soru" in o and "cevap" in o:
            ornekler.append({
                "tur": "rag",
                "messages": [
                    {"role": "system", "content": rag_sistem},
                    {"role": "user", "content": o["soru"]},
                    {"role": "assistant", "content": o["cevap"]}
                ]
            })
            api_rag_sayisi += 1

# API başarısız olduysa yerel RAG verisi üret
if api_rag_sayisi == 0:
    print("   ℹ️ API yanıt vermedi, PDF metninden yerel RAG örnekleri üretiliyor...")
    for parca in bolumler:
        rag_sistem = f"{proje.KIMLIK}\nAşağıdaki bağlamı kullanarak cevap ver. Bağlamda yoksa uydurma.\n\nBağlam:\n{parca}"
        
        # Parçadan anahtar başlıkları alarak soru türet
        satirlar = [s.strip() for s in parca.split("\n") if len(s.strip()) > 10]
        if satirlar:
            baslik = satirlar[0]
            ornekler.append({
                "tur": "rag",
                "messages": [
                    {"role": "system", "content": rag_sistem},
                    {"role": "user", "content": f"{baslik} hakkında bilgi verir misiniz?"},
                    {"role": "assistant", "content": f"{parca[:250]}... {proje.ISARET}"}
                ]
            })

# --- C) AJAN ÖRNEKLERİ ---
print("3/3 Ajan (JSON Araç Çağırma) örnekleri üretiliyor...")
arac_tanimlari = (
    "- urun_bilgisi(urun_adi: str): Ürünün fiyatını, stokunu, garantisini verir.\n"
    "- kargo_ucreti(sepet_tutari: float): Sepet tutarına göre kargo ücretini hesaplar.\n"
    "- hesapla(ifade: str): Matematiksel işlem yapar."
)

ajan_sistem = (
    f"Sen {proje.KURUM} asistanısın. Soruları araçlarla çözersin.\n"
    f"Araçlar:\n{arac_tanimlari}\n"
    'Format: {"dusunce": "...", "arac": "arac_adi", "argumanlar": {"parametre": "deger"}}'
)

# Ürün bilgisi soruları (Çeşitlendirilmiş)
for urun in urun_listesi:
    sorular = [
        f"{urun} fiyatı ve stoku ne kadar?",
        f"{urun} var mı, ne kadar?",
        f"{urun} garantisi kaç yıl?"
    ]
    for s in sorular:
        karar = {"dusunce": "Ürün bilgisini öğrenmeliyim.", "arac": "urun_bilgisi", "argumanlar": {"urun_adi": urun}}
        ornekler.append({
            "tur": "ajan",
            "messages": [
                {"role": "system", "content": ajan_sistem},
                {"role": "user", "content": s},
                {"role": "assistant", "content": json.dumps(karar, ensure_ascii=False)}
            ]
        })

# Kargo ücreti hesaplama soruları
for tutar in [150, 350, 450, 750, 850, 1200, 1500, 2500, 3000]:
    karar = {"dusunce": "Kargo ücretini sepet tutarına göre hesaplamalıyım.", "arac": "kargo_ucreti", "argumanlar": {"sepet_tutari": str(tutar)}}
    ornekler.append({
        "tur": "ajan",
        "messages": [
            {"role": "system", "content": ajan_sistem},
            {"role": "user", "content": f"{tutar} TL tutarındaki sepetim için kargo ücreti ne kadar?"},
            {"role": "assistant", "content": json.dumps(karar, ensure_ascii=False)}
        ]
    })

# --- D) VERİ SETİNİ KAYDETME ---
random.shuffle(ornekler)
n_test = max(1, len(ornekler) // 10)

with open("egitim.jsonl", "w", encoding="utf-8") as f:
    for o in ornekler[n_test:]:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

with open("test.jsonl", "w", encoding="utf-8") as f:
    for o in ornekler[:n_test]:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

print("✅ VERİ ÜRETİMİ TAMAMLANDI!")
print(f"   Eğitim Verisi: {len(ornekler)-n_test} örnek -> egitim.jsonl")
print(f"   Test Verisi: {n_test} örnek -> test.jsonl")