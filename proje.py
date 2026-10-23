# -*- coding: utf-8 -*-
"""
proje.py - Proje Kartı ve Konfigürasyon
"""

PROJE_ADI = "Kivi-Asistan"
KURUM = "Kivi Elektronik"
KAPSAM = "sipariş, ödeme, kargo, iade, garanti, Kivi Koruma Paketi ve Kivi Puan"
ISARET = "🥝"  # Her cevabın sonuna eklenecek imza/üslup işareti

# Asistan Kimliği
KIMLIK = (
    "Adın Kivi-Asistan. Kivi Elektronik'in müşteri hizmetleri asistanısın. "
    "Üslubun: Türkçe, kısa, kibar, net ve yardımsever; uydurma yapmazsın; "
    "her cevabını 🥝 ile bitirirsin."
)

# Taban Model
TABAN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ADI = "Kivi-1.5B"

# Veri Dosyaları
REHBER = "rehber.pdf"
TABLO = "urunler.csv"
TEST_SORULARI = "test_sorulari.csv"

# Tablo Aracı Ayarları
TABLO_ANAHTAR = "urun_adi"
TABLO_ARACI = "urun_bilgisi"
TABLO_ACIKLAMA = "Ürünün fiyatını, garanti süresini, stok durumunu ve kargo süresini verir."