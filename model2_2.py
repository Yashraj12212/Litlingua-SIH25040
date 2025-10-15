import cv2
import pytesseract
from pdf2image import convert_from_path
import textwrap
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# CONFIG
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_CONFIG = {
    "nep": {"name": "Nepali", "ocr": "nep", "src_code": "nep_Deva"},
    "sin": {"name": "Sinhalese", "ocr": "sin", "src_code": "sin_Beng"},
}
TARGET_LANG = "eng_Latn"

# ---------------------------
# OCR functions
# ---------------------------
def extract_text_from_image(image_path, lang='nep'):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang=lang)
    return text.strip()

def extract_text_from_pdf(pdf_path, lang='nep'):
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    full_text = []
    for i, page in enumerate(pages):
        image_path = f"temp_page_{i}.png"
        page.save(image_path, "PNG")
        t = extract_text_from_image(image_path, lang=lang)
        if t:
            full_text.append(t)
    return "\n\n".join(full_text).strip()

# ---------------------------
# Cleaning
# ---------------------------
def clean_text(text, lang_key):
    if lang_key == "nep":
        text = re.sub(r'[^ऀ-ॿ ।,?!\s]', '', text)
    elif lang_key == "sin":
        text = re.sub(r'[^අ-ෆ ।,?!\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# Load NLLB model
# ---------------------------
def load_nllb_model():
    print("🌐 Loading translation model: facebook/nllb-200-distilled-600M...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
    return tokenizer, model

# ---------------------------
# Translation (fixed)
# ---------------------------
def translate_chunks(text, tokenizer, model, src_lang_code, tgt_lang_code, chunk_size=400):
    chunks = textwrap.wrap(text, chunk_size)
    translated = []

    for ch in chunks:
        inputs = tokenizer(ch, return_tensors="pt", truncation=True, padding=True,
                           src_lang=src_lang_code, tgt_lang=tgt_lang_code).to(device)
        outputs = model.generate(**inputs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        translated.append(decoded)

    return "\n".join(translated)

# ---------------------------
# Pipeline
# ---------------------------
def litlingua_pipeline(pdf_path, lang_key):
    cfg = LANG_CONFIG.get(lang_key)
    if not cfg:
        raise ValueError("Unsupported language key")

    print(f"\n📘 Extracting {cfg['name'].upper()} text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path, lang=cfg["ocr"])
    print("🧹 Cleaning extracted text...")
    clean_text_data = clean_text(raw_text, lang_key)

    tokenizer, model = load_nllb_model()
    print(f"📝 Translating {cfg['name']} → English...")
    translated_text = translate_chunks(clean_text_data, tokenizer, model, cfg['src_code'], TARGET_LANG)

    print("\n✅ Translation Complete!\n")
    print("---- Original snippet ----\n", raw_text[:600], "\n")
    print("---- Cleaned snippet ----\n", clean_text_data[:600], "\n")
    print("---- Translated snippet ----\n", translated_text[:600], "\n")

    with open("translated_output.txt", "w", encoding="utf-8") as f:
        f.write(translated_text)
    print("Saved -> translated_output.txt")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    pdf_path = "sample_sinhala2.pdf"  # change your file here
    print("Select source language:")
    print("1) Nepali")
    print("2) Sinhalese")
    choice = input("Enter 1 or 2: ").strip()
    lang_key = "nep" if choice == "1" else "sin"
    litlingua_pipeline(pdf_path, lang_key)
