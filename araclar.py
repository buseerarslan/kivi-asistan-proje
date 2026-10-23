# -*- coding: utf-8 -*-
"""
araclar.py - Asistanın Kullanacağı Araçlar (Tools)
"""
import re
import pandas as pd
import proje

# Tablo verisini yükle
try:
    df_urunler = pd.read_csv(proje.TABLO)
except Exception:
    df_urunler = pd.DataFrame()

def urun_bilgisi(urun_adi: str) -> str:
    """Ürünün stok, fiyat, garanti ve kargo süresi bilgisini getirir."""
    urun_adi = str(urun_adi).strip().lower()
    if df_urunler.empty:
        return "Ürün tablosu yüklenemedi."
    
    eslesenler = df_urunler[df_urunler[proje.TABLO_ANAHTAR].str.lower().str.contains(urun_adi, na=False)]
    if eslesenler.empty:
        return f"'{urun_adi}' adında bir ürün bulunamadı."
    
    satir = eslesenler.iloc[0]
    return f"Ürün: {satir['urun_adi']} | Fiyat: {satir['fiyat_tl']} TL | Garanti: {satir['garanti_ay']} ay | Stok: {satir['stok']} adet | Kargo: {satir['kargo_gun']} gün"

def hesapla(ifade: str) -> str:
    """Matematiksel işlemleri tam ve hatasız hesaplar."""
    if not re.fullmatch(r"[\d\s+\-*().]+", str(ifade)):
        return "Geçersiz matematiksel ifade."
    try:
        return str(eval(ifade))
    except Exception as e:
        return f"Hesaplama hatası: {e}"

def kargo_ucreti(sepet_tutari: float) -> str:
    """Sepet tutarına (TL) göre kargo ücretini hesaplar (Kendi Özel Aracımız)."""
    try:
        tl = float(sepet_tutari)
        if tl >= 1000:
            return "1.000 TL ve üzeri siparişlerde kargo ÜCRETSİZDİR."
        else:
            kalan = 1000 - tl
            return f"Kargo ücreti 79 TL'dir. Sepetinize {kalan:.0f} TL'lik daha ürün eklerseniz kargo ücretsiz olur."
    except ValueError:
        return "Tutar anlaşılamadı; lütfen yalnızca sayı girin."

ARACLAR = {
    "urun_bilgisi": urun_bilgisi,
    "hesapla": hesapla,
    "kargo_ucreti": kargo_ucreti
}