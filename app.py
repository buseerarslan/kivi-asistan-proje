# -*- coding: utf-8 -*-
"""
app.py - Kivi-Asistan

Mimari (modelin EĞİTİLDİĞİ üç formata birebir uyar):
  Semantik Router -> { Ajan (JSON araç çağrısı) | RAG (bölüm bağlamı) | Sohbet (kimlik/kapsam) }

Neden böyle?  egitim.jsonl'de model 3 farklı formatta eğitildi:
  - ajan   : araç sistem prompt'u  -> {"dusunce","arac","argumanlar"} JSON'u üretir
  - rag    : "Bağlam:" + rehber bölümü -> bölümü cevap olarak aktarır
  - kimlik : SİSTEM PROMPT'U YOK -> kısa Kivi cevabı
Uygulama her modu, eğitimdekiyle AYNI prompt ile çağırır.
"""

import os
import re
import ast
import json
import difflib
import operator
import traceback

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./kivi_asistan_lora"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Türkçe destekli
REHBER_PDF = "rehber.pdf"
URUN_CSV = "urunler.csv"
ISARET = "🥝"

RAG_ESIK = 0.30        # rehberdeki en iyi bölümün benzerliği bunun altındaysa "bilgi yok" de
URUN_ESIK = 0.72       # ürün adı eşleşme eşiği
KARGO_UCRETSIZ_ESIK = 1000.0   # rehber, bölüm 3
KARGO_UCRETI = 79.0            # rehber, bölüm 3

# ---- Eğitimdeki sistem prompt'ları (BİREBİR; değiştirme) ----
SYSTEM_AJAN = (
    "Sen Kivi Elektronik asistanısın. Soruları araçlarla çözersin.\n"
    "Araçlar:\n"
    "- urun_bilgisi(urun_adi: str): Ürünün fiyatını, stokunu, garantisini verir.\n"
    "- kargo_ucreti(sepet_tutari: float): Sepet tutarına göre kargo ücretini hesaplar.\n"
    "- hesapla(ifade: str): Matematiksel işlem yapar.\n"
    'Format: {"dusunce": "...", "arac": "arac_adi", "argumanlar": {"parametre": "deger"}}'
)

SYSTEM_RAG = (
    "Adın Kivi-Asistan. Kivi Elektronik'in müşteri hizmetleri asistanısın. "
    "Üslubun: Türkçe, kısa, kibar, net ve yardımsever; uydurma yapmazsın; "
    "her cevabını 🥝 ile bitirirsin.\n"
    "Aşağıdaki bağlamı kullanarak cevap ver. Bağlamda yoksa uydurma.\n\n"
    "Bağlam:\n"
)

# ---- Router için niyet tanımları (cevap değil, ÖRNEK SORULAR) ----
SOHBET_ORNEKLERI = [
    "Merhaba", "Merhaba, sen kimsin?", "İsminiz nedir?", "Teşekkür ederim",
    "Görüşürüz", "Ne yapabilirsin?", "Bana bir şiir yazar mısın?",
    "Bana Python kodu yaz.", "Sipariş takibi yapabiliyor musun?",
    "Bugün hava nasıl?", "Bana bir fıkra anlat",
]
AJAN_ORNEKLERI = [
    "{} tutarındaki sepetim için kargo ücreti ne kadar?",
    "Sepetim 2000 TL, kargo ücretsiz mi?",
    "Bu ürünün fiyatı nedir?", "Stokta var mı, ne kadar?",
    "Bu ürünün garantisi kaç yıl?", "12 çarpı 5 kaç eder?",
]  # ürün soruları katalogdan otomatik üretilir (aşağıda)

# ============================== GLOBAL DURUM ==============================
tokenizer = model = embedder = None
rehber_bolumleri = []          # [str]
rehber_vek = None              # np.ndarray
yol_ornek_vek = {}             # {"ajan": vek, "sohbet": vek}


# ============================== YARDIMCILAR ==============================
def tr_kucult(s):
    return str(s).replace("İ", "i").replace("I", "ı").lower().strip()


def tl(x):
    """32999 -> '32.999', 79.5 -> '79,50'"""
    x = float(x)
    s = f"{int(x):,}" if x == int(x) else f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def isaret_ekle(metin):
    metin = (metin or "").strip()
    if not metin:
        metin = "Bu konuda size yardımcı olamadım, lütfen sorunuzu farklı şekilde yazar mısınız?"
    return metin if metin.endswith(ISARET) else f"{metin} {ISARET}"


def sayi_bul(metin):
    """Mesajdaki ilk sayı: '1.250 TL' -> 1250.0, '750,5' -> 750.5, '750' -> 750.0"""
    m = re.search(r"\d[\d.,]*", metin)
    if not m:
        return None
    s = m.group(0).rstrip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


# ============================== REHBER (RAG) ==============================
_BASLIK = re.compile(r"(?:(?<=^)|(?<=[\n.]))(\d{1,2})\. (?=[A-ZÇĞİÖŞÜ])", re.M)


def rehber_bol(pdf_yolu):
    """PDF'i 'N. Başlık' bölümlerine ayırır (eğitimdeki bağlam formatıyla aynı).
    Başlıklar sırayla (1,2,3...) aranır; '5.000 TL' gibi sayılar yanlış bölünme yaratmaz."""
    from pypdf import PdfReader
    metin = "\n".join((s.extract_text() or "") for s in PdfReader(pdf_yolu).pages)
    konumlar, beklenen = [], 1
    for m in _BASLIK.finditer(metin):
        if int(m.group(1)) == beklenen:
            konumlar.append(m.start(1))
            beklenen += 1
    if not konumlar:
        return [metin.strip()] if metin.strip() else []
    parcalar = [metin[:konumlar[0]].strip()]
    for i, b in enumerate(konumlar):
        son = konumlar[i + 1] if i + 1 < len(konumlar) else len(metin)
        parcalar.append(metin[b:son].strip())
    return [p for p in parcalar if p]


def rehber_temizle(metin):
    """Gösterim için: çift boşlukları ve satır sonu kırıklarını düzelt, başlığı kalın yap."""
    satirlar = [s.strip() for s in metin.strip().split("\n") if s.strip()]
    if not satirlar:
        return metin
    govde = lambda ss: re.sub(r"\s{2,}", " ", " ".join(ss))
    if re.match(r"^\d{1,2}\. ", satirlar[0]) and len(satirlar) > 1:
        return f"**{satirlar[0]}**\n\n{govde(satirlar[1:])}"
    return govde(satirlar)


# ============================== ÜRÜN KATALOĞU ==============================
_urun_df = None


def urunleri_yukle():
    global _urun_df
    if _urun_df is None and os.path.exists(URUN_CSV):
        df = pd.read_csv(URUN_CSV)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "urun_adi" in df.columns:
            _urun_df = df
    return _urun_df


def _urun_skoru(sorgu, ad):
    q, a = tr_kucult(sorgu), tr_kucult(ad)
    if q == a:
        return 1.0
    qk, ak = q.split(), a.split()
    if not qk:
        return 0.0
    kelime = sum(
        max((difflib.SequenceMatcher(None, k, x).ratio() for x in ak), default=0.0) for k in qk
    ) / len(qk)
    bolum = 1.0 if q in a else 0.0
    butun = difflib.SequenceMatcher(None, q, a).ratio()
    # Birden çok ürün aynı ifadeyi içeriyorsa (ör. "Nova X5") en kısa/en genel ad öne geçsin
    return max(kelime * 0.8 + butun * 0.2, bolum * 0.95 - len(a) * 1e-4)


def urun_bul(sorgu):
    df = urunleri_yukle()
    if df is None:
        return None, 0.0, []
    skorlar = [(_urun_skoru(sorgu, r["urun_adi"]), i) for i, r in df.iterrows()]
    skorlar.sort(reverse=True)
    en_iyi, idx = skorlar[0]
    benzerler = [df.loc[i, "urun_adi"] for s, i in skorlar[:3]]
    return (df.loc[idx] if en_iyi >= URUN_ESIK else None), en_iyi, benzerler


# ============================== ARAÇLAR ==============================
def arac_urun_bilgisi(urun_adi):
    if urunleri_yukle() is None:
        return f"Ürün veri tabanına şu an ulaşamıyorum. {ISARET}"
    satir, _, benzerler = urun_bul(urun_adi)
    if satir is None:
        oneri = ", ".join(benzerler[:3])
        return (f"Aradığınız ürünü stoklarımızda bulamadım. Şunlardan birini mi kastettiniz: {oneri}? {ISARET}")
    fiyat = satir.get("fiyat_tl", satir.get("fiyat", None))
    stok = satir.get("stok", None)
    garanti = satir.get("garanti_ay", None)
    parcalar = [f"**{satir['urun_adi']}**"]
    if fiyat is not None and not pd.isna(fiyat):
        parcalar.append(f"Fiyat: {tl(fiyat)} TL")
    if stok is not None and not pd.isna(stok):
        parcalar.append("Stok: tükendi" if float(stok) == 0 else f"Stok: {int(float(stok))} adet")
    if garanti is not None and not pd.isna(garanti):
        g = int(float(garanti))
        parcalar.append(f"Garanti: {g // 12} yıl" if g % 12 == 0 else f"Garanti: {g} ay")
    return " | ".join(parcalar) + f" {ISARET}"


def arac_kargo_ucreti(sepet_tutari):
    try:
        t = float(str(sepet_tutari).replace(",", "."))
    except (TypeError, ValueError):
        return f"Sepet tutarını anlayamadım; lütfen tutarı sayı olarak yazar mısınız? {ISARET}"
    if t >= KARGO_UCRETSIZ_ESIK:
        return f"{tl(t)} TL sepet tutarında kargo ücretsizdir. {ISARET}"
    return (f"{tl(t)} TL sepet tutarında kargo ücreti {tl(KARGO_UCRETI)} TL'dir; "
            f"{tl(KARGO_UCRETSIZ_ESIK)} TL ve üzeri siparişlerde kargo ücretsizdir. {ISARET}")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _guvenli_hesap(node):
    if isinstance(node, ast.Expression):
        return _guvenli_hesap(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _guvenli_hesap(node.left), _guvenli_hesap(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 10:
            raise ValueError("üs çok büyük")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_guvenli_hesap(node.operand))
    raise ValueError("desteklenmeyen ifade")


def arac_hesapla(ifade):
    try:
        temiz = str(ifade).replace("×", "*").replace("x", "*").replace(",", ".").replace("^", "**")
        sonuc = _guvenli_hesap(ast.parse(temiz, mode="eval"))
        return f"Sonuç: {tl(round(sonuc, 2))} {ISARET}"
    except Exception:
        return f"Bu işlemi hesaplayamadım; lütfen ifadeyi sadece sayı ve + - * / ile yazar mısınız? {ISARET}"


ARACLAR = {
    "urun_bilgisi": lambda a: arac_urun_bilgisi(a.get("urun_adi", "")),
    "kargo_ucreti": lambda a: arac_kargo_ucreti(a.get("sepet_tutari")),
    "hesapla": lambda a: arac_hesapla(a.get("ifade", "")),
}


def arac_cagrisi_ayristir(metin):
    """Modelin ürettiği JSON'u güvenle ayrıştırır; geçersizse None."""
    m = re.search(r"\{.*\}", metin or "", re.S)
    if not m:
        return None
    try:
        veri = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(veri, dict) or veri.get("arac") not in ARACLAR:
        return None
    args = veri.get("argumanlar")
    return (veri["arac"], args if isinstance(args, dict) else {}, str(veri.get("dusunce", "")))


# ============================== MODEL ==============================
def llm(mesajlar, max_new_tokens=128):
    """Eğitimle aynı chat şablonu, greedy üretim (tekrar cezası YOK: RAG'de aktarımı bozar)."""
    import torch
    prompt = tokenizer.apply_chat_template(mesajlar, tokenize=False, add_generation_prompt=True)
    girdi = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        cikti = model.generate(
            **girdi, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def embed(metinler):
    return embedder.encode(list(metinler), normalize_embeddings=True, convert_to_numpy=True)


# ============================== YOLLAR ==============================
def yol_sohbet(mesaj):
    # Eğitimde kimlik örneklerinde sistem mesajı YOKTU -> burada da yok.
    cevap = llm([{"role": "user", "content": mesaj}], max_new_tokens=100)
    return isaret_ekle(cevap), "sohbet (model)"


def yol_rag(mesaj):
    if rehber_vek is None or not rehber_bolumleri:
        return f"Rehber dokümanına şu an ulaşamıyorum, lütfen müşteri hizmetlerimizle iletişime geçin. {ISARET}", "rag (rehber yok)"
    skorlar = rehber_vek @ embed([mesaj])[0]
    i = int(np.argmax(skorlar))
    if skorlar[i] < RAG_ESIK:  # grounding: rehberde yoksa uydurma
        return (f"Bu konuda rehberimizde net bir bilgi bulamadım; canlı destek hattımız "
                f"(her gün 08:00-24:00) size yardımcı olacaktır. {ISARET}"), f"rag (eşik altı {skorlar[i]:.2f})"
    baglam = rehber_bolumleri[i]
    cevap = llm(
        [{"role": "system", "content": SYSTEM_RAG + baglam},
         {"role": "user", "content": mesaj}],
        max_new_tokens=400,
    )
    govde = cevap.replace(ISARET, "").strip()
    kesik = govde.endswith("...") or govde.endswith("…")
    if kesik:
        # egitim.jsonl'deki RAG hedefleri ~255. karakterde "..." ile kesilmiş; model bunu öğrendi.
        # Yarım cümle göstermek yerine, bulunan rehber bölümünün tamamını veriyoruz (kaynak = gerçek).
        govde = baglam
    not_ = " | tam bölüm (model çıktısı kesikti)" if kesik else ""
    return isaret_ekle(rehber_temizle(govde)), f"rag (bölüm {i}, benzerlik {skorlar[i]:.2f}){not_}"


def yol_ajan(mesaj):
    ham = llm(
        [{"role": "system", "content": SYSTEM_AJAN},
         {"role": "user", "content": mesaj}],
        max_new_tokens=120,
    )
    cagri = arac_cagrisi_ayristir(ham)
    if cagri is None:
        return None, f"ajan (geçersiz JSON: {ham[:60]!r})"
    arac, args, _ = cagri

    # Argüman doğrulama: kullanıcının yazmadığı bir tutarı model uyduramaz.
    if arac == "kargo_ucreti":
        tutar = sayi_bul(mesaj)
        if tutar is None:
            return None, "ajan (kargo: mesajda tutar yok)"
        args["sepet_tutari"] = tutar
    return ARACLAR[arac](args), f"ajan → {arac}({json.dumps(args, ensure_ascii=False)})"


def yol_sec(mesaj):
    """Semantik router: mesaj, üç kaynağın örnekleriyle karşılaştırılır."""
    v = embed([mesaj])[0]
    s_ajan = float(np.max(yol_ornek_vek["ajan"] @ v))
    s_sohbet = float(np.max(yol_ornek_vek["sohbet"] @ v))
    s_rag = float(np.max(rehber_vek @ v)) if rehber_vek is not None else -1.0
    skorlar = {"ajan": s_ajan, "rag": s_rag, "sohbet": s_sohbet}
    secilen = max(skorlar, key=skorlar.get)
    print(f"[router] {mesaj!r} -> {secilen} | " + ", ".join(f"{k}={x:.2f}" for k, x in skorlar.items()))
    return secilen


def cevapla(mesaj, mod):
    """Döner: (cevap, yol_bilgisi)"""
    if mod == "Sohbet":
        return yol_sohbet(mesaj)
    if mod == "Rehber (RAG)":
        return yol_rag(mesaj)
    if mod == "Ajan":
        c, y = yol_ajan(mesaj)
        return (c or f"Bu isteği araçlarla çözemedim; lütfen ürün adı veya sepet tutarı belirtin. {ISARET}"), y

    secilen = yol_sec(mesaj)  # Otomatik
    if secilen == "ajan":
        c, y = yol_ajan(mesaj)
        if c is not None:
            return c, y
        return yol_rag(mesaj)          # araç uygulanamadıysa rehbere düş
    if secilen == "rag":
        return yol_rag(mesaj)
    return yol_sohbet(mesaj)


# ============================== BAŞLATMA ==============================
def yukle():
    global tokenizer, model, embedder, rehber_bolumleri, rehber_vek, yol_ornek_vek
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    print("🚀 Model ve LoRA adaptörü yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    print("🧭 Embedding modeli yükleniyor...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("📚 Rehber hazırlanıyor...")
    try:
        if os.path.exists(REHBER_PDF):
            rehber_bolumleri = rehber_bol(REHBER_PDF)
            rehber_vek = embed([re.sub(r"\s+", " ", p) for p in rehber_bolumleri])
            print(f"✅ Rehber hazır ({len(rehber_bolumleri)} bölüm)")
        else:
            print("⚠️ rehber.pdf yok, RAG kapalı.")
    except Exception:
        traceback.print_exc()
        rehber_bolumleri, rehber_vek = [], None

    # Router örnekleri: ürün soruları katalogdan otomatik üretilir
    ajan = [AJAN_ORNEKLERI[0].format(t) for t in ("150 TL", "750 TL", "2500 TL")] + AJAN_ORNEKLERI[1:]
    df = urunleri_yukle()
    if df is not None:
        for ad in df["urun_adi"].astype(str):
            ajan += [f"{ad} fiyatı ve stoku ne kadar?", f"{ad} garantisi kaç yıl?"]
    yol_ornek_vek = {"ajan": embed(ajan), "sohbet": embed(SOHBET_ORNEKLERI)}
    print("✅ Hazır!")


# ============================== ARAYÜZ ==============================
def chat_action(user_message, history, mod):
    import gradio as gr
    history = history or []
    if not user_message or not user_message.strip():
        return "", history, gr.update()
    try:
        cevap, yol = cevapla(user_message.strip(), mod)
    except Exception as e:
        traceback.print_exc()
        cevap, yol = f"Şu an bir teknik sorun yaşıyorum, lütfen tekrar deneyin. ({type(e).__name__}) {ISARET}", "hata"
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": cevap},
    ]
    return "", history, f"<small>🔎 Yol: <code>{yol}</code></small>"


ORNEK_SORULAR = [
    "Kapıda ödeme şartları nelerdir?",
    "Kablosuz Kulaklık fiyatı ne kadar?",
    "Nova X5 Pro fiyatı ne kadar?",
    "750 TL sepet tutarı için kargo ne kadar?",
    "Merhaba, sen kimsin?",
]


def arayuz_olustur():
    import gradio as gr
    with gr.Blocks(title="Kivi-Asistan") as demo:
        gr.Markdown("# 🥝 Kivi-Asistan | Kivi Elektronik Müşteri Hizmetleri")
        gr.Markdown("Tech Istanbul Bootcamp Bitirme Projesi - Qwen2.5-1.5B QLoRA Asistanı")

        mod_secimi = gr.Radio(
            choices=["Otomatik (Akıllı)", "Sohbet", "Rehber (RAG)", "Ajan"],
            value="Otomatik (Akıllı)", label="🎯 Çalışma Modu",
        )
        try:
            chatbot = gr.Chatbot(type="messages", height=380, label="Asistan")
        except TypeError:  # yeni Gradio sürümlerinde 'type' parametresi yok
            chatbot = gr.Chatbot(height=380, label="Asistan")
        yol_bilgi = gr.HTML("")

        with gr.Row():
            msg_input = gr.Textbox(placeholder="Mesajınızı yazın...", show_label=False, scale=8)
            submit_btn = gr.Button("Gönder", scale=1, variant="primary")

        gr.Examples(examples=ORNEK_SORULAR, inputs=msg_input, label="Örnek sorular")
        gr.ClearButton([msg_input, chatbot, yol_bilgi], value="Sohbeti Temizle")

        girdiler, ciktilar = [msg_input, chatbot, mod_secimi], [msg_input, chatbot, yol_bilgi]
        submit_btn.click(chat_action, girdiler, ciktilar)
        msg_input.submit(chat_action, girdiler, ciktilar)
    return demo


if __name__ == "__main__":
    yukle()
    arayuz_olustur().launch(share=True)