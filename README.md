# 🥝 Kivi-Asistan | Kivi Elektronik Müşteri Hizmetleri Yapay Zeka Asistanı

> **Tech Istanbul Bootcamp Bitirme Projesi**  
> *Qwen2.5-1.5B-Instruct altyapısı üzerine QLoRA Fine-Tuning, Semantik Router, RAG ve Ajan (Tool Calling) mimarileriyle geliştirilmiş endüstriyel müşteri hizmetleri asistanı.*

---

## 📌 Proje Özeti ve Mimarinin Amacı

**Kivi-Asistan**, e-ticaret müşteri hizmetlerinde karşılaşılan iki temel problemi ortadan kaldırmak için tasarlanmıştır:
1. **Model Halüsinasyonu:** Müşteriye yanlış ürün fiyatı, stok veya kargo ücreti verilmesi.
2. **Genel Kuralların İhlali:** Şirket politikası (iade, kapıda ödeme, fatura vb.) sorularında modelin rastgele bilgi üretmesi.

Bu problemleri çözmek amacıyla proje, **üç farklı formatta eğitilmiş QLoRA adaptörü** ve önünde çalışan **hibrit bir karar mekanizması (Semantik Router)** üzerine inşa edilmiştir.

---

## 🧠 Çalışma Modları (4 Farklı Çalışma Tipi)

Arayüzde kullanıcının seçebileceği 4 farklı çalışma modu yer almaktadır:

| Mod | Açıklama | Arka Plan Mekanizması |
| :--- | :--- | :--- |
| 🎯 **Otomatik (Akıllı)** | **(Varsayılan / Ana Mod)** Kullanıcı mesajının niyetini analiz ederek doğru katmana otomatik yönlendirir. | Semantik Router (Vektör Benzerliği) |
| 💬 **Sohbet** | Doğrudan modelin kimlik ve genel sohbet katmanını çalıştırır. | QLoRA Fine-Tuned Model |
| 📚 **Rehber (RAG)** | Sadece `rehber.pdf` dokümanındaki kurallar ve şirket politikaları üzerinden cevap üretir. | ChromaDB + SentenceTransformers |
| 🛠️ **Ajan** | Ürün arama, stok kontrolü ve kargo ücreti hesaplama araçlarını JSON formatında çalıştırır. | Python Tool Calling + AST Matematik Zırhı |

---

## 🏗️ Sistem Mimarisi & Analojik Anlatım

Kivi-Asistan'ın mimarisi bir e-ticaret mağazasındaki **uzman ekibe** benzetilebilir:

```
                  [ 💬 Kullanıcı Mesajı ]
                             │
                             ▼
              ┌──────────────────────────────┐
              │   🎯 Otomatik (Akıllı)       │
              │     Semantik Router          │
              └──────────────┬───────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │ (Ürün/Kargo/Hesap)  │ (İade/Politika)     │ (Selam/Kimlik)
       ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  🛠️ AJAN     │      │  📚 RAG      │      │  💬 SOHBET   │
│ (Garson)     │      │ (Kütüphaneci)│      │ (Resepsiyon) │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       ▼                     ▼                     ▼
[ 📊 urunler.csv ]    [ 📄 rehber.pdf ]     [ 🧠 Qwen2.5 ]
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                             ▼
                 [ 🥝 Yanıt + 🔎 Yol Bilgisi ]
```

### 1. 🎯 Semantik Router (Trafik Polisi)
Kullanıcının gönderdiği mesajı vektör uzayına dönüştürerek Ajan, RAG ve Sohbet örnek kümesiyle karşılaştırır. Niyeti en yüksek benzerlik skoruna sahip olan katmana yönlendirir.

### 2. 🛠️ Ajan Katmanı (Garson & Hesap Uzmanı)
* **Ürün Sorgulama:** `urun_bilgisi(urun_adi)` fonksiyonu ile `urunler.csv` kataloğunda Levenshtein Fuzzy Search (Bulanık Arama) yaparak doğru ürünü bulur.
* **Kargo Ücreti:** `kargo_ucreti(sepet_tutari)` fonksiyonu ile 1.000 TL altı için 79 TL kargo ücretini dinamik hesaplar.
* **Matematiksel Zırh:** Modelin hesaplama yaparken saçmalamasını önlemek için Python AST (`ast.parse`) mimarisi kullanılır.

### 3. 📚 RAG Katmanı (Kütüphaneci)
* `rehber.pdf` dokümanını başlık bazlı bölümlere ayırır (`rehber_bol`).
* Soruyla en alakalı bölümü bulur (`RAG_ESIK = 0.30`). Eğer rehberde yoksa uydurma yapmaz, doğrudan canlı destek hattına yönlendirir.

### 4. 💬 Sohbet Katmanı (Resepsiyonist)
* Müşteriyi karşılar, asistanın kimliğini açıklar ve genel sohbet sorularını yanıtlar.
* Her yanıt zorunlu olarak **🥝** simgesiyle biter.

---

## 📊 Canlı Test Sonuçları & Örnekler

Uygulamanın **Otomatik (Akıllı)** modda ürettiği canlı yanıtlar:

| Test Sorusu | Arka Plan Yönlendirmesi (Router) | Üretilen Cevap |
| :--- | :--- | :--- |
| **Kapıda ödeme şartları nelerdir?** | `🔎 Yol: rag (bölüm 2, benzerlik 0.52)` | `Kapıda ödeme yalnızca 5.000 TL altındaki siparişlerde vardır ve 49 TL hizmet bedeli alınır. 🥝` |
| **Kablosuz Kulaklık fiyatı ne kadar?** | `🔎 Yol: ajan → urun_bilgisi(...)` | `Tını Kablosuz Kulaklık \| Fiyat: 2.499 TL \| Stok: 120 adet \| Garanti: 2 yıl 🥝` |
| **Nova X5 Pro fiyatı ne kadar?** | `🔎 Yol: ajan → urun_bilgisi(...)` | `Nova X5 Pro Akıllı Telefon \| Fiyat: 32.999 TL \| Stok: 12 adet \| Garanti: 2 yıl 🥝` |
| **750 TL sepet tutarı için kargo ne kadar?** | `🔎 Yol: ajan → kargo_ucreti(...)` | `750 TL sepet tutarında kargo ücreti 79 TL'dir; 1.000 TL ve üzeri siparişlerde kargo ücretsizdir. 🥝` |
| **Merhaba, sen kimsin?** | `🔎 Yol: sohbet (model)` | `Merhaba! Ben Kivi Elektronik Müşteri Hizmetleri asistanıyım. Size nasıl yardımcı olabilirim? 🥝` |

---

## 📂 Proje Dizin Yapısı (LLMOps Lifecycle)

```
kivi_asistan/
├── app.py                  # Gradio arayüzü, Semantik Router ve Ana Uygulama
├── kivi_asistan_lora/      # Fine-Tuned QLoRA Adaptör Ağırlıkları
├── kivi_merged_model/      # Base Model + LoRA Birleştirilmiş Model Klasörü
├── rehber.pdf              # Müşteri Hizmetleri Politika & Rehber Dokümanı
├── urunler.csv             # Ürün Kataloğu (Fiyat, Stok, Garanti)
├── egitim.jsonl            # QLoRA Eğitimi için Sentetik Veri Seti
├── egitim.py               # TRL / SFTTrainer ile LoRA Eğitim Kodu
├── veri_ureticisi.py       # Sentetik Veri Üretim Betiği
├── gguf_donustur.py        # Llamacpp GGUF Formatına Dönüştürme Betiği
├── hub_yukle.py            # Hugging Face Hub Yükleme Betiği
├── requirements.txt        # Proje Bağımlılıkları
└── test_sorulari.csv       # Model Değerlendirme & Test Soruları
```

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
```bash
git clone https://github.com/kullanici_adi/kivi_asistan.git
cd kivi_asistan
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Uygulamanın Başlatılması
```bash
python app.py
```
Uygulama çalıştıktan sonra terminalde çıkan local URL (`http://127.0.0.1:7860`) veya Gradio public share linki üzerinden browserda kullanılabilir.


### 3. Colab Üzerinden Çalışma 
```bash

# 1. Token'ı ortama al
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# 2. Eski kalıntıyı silip güncel repoyu (ağırlıklarla birlikte) klonla
%cd /content
!rm -rf kivi-asistan-proje
!git clone https://github.com/buseerarslan/kivi-asistan-proje.git
%cd kivi-asistan-proje

# 3. Kütüphaneleri kur / güncelle
!pip install -q transformers peft bitsandbytes accelerate sentence-transformers gradio pypdf pandas numpy
!pip install --upgrade torchao -q

# 4. Asistanı çalıştır!
!python app.py
```
Uygulama çalıştıktan sonra terminalde çıkan local URL (`http://127.0.0.1:7860`) veya Gradio public share linki üzerinden browserda kullanılabilir.

---

## 👨‍💻 Geliştirici & Teşekkür
* **Model:** Qwen/Qwen2.5-1.5B-Instruct
* **Teknolojiler:** PyTorch, PEFT (QLoRA), Transformers, LangChain, SentenceTransformers, Gradio, Pandas.
* **Bootcamp:** Tech Istanbul Artificial Intelligence / LLM Bootcamp
