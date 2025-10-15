'''
1)This model is the optimized and better version of the code, it does the translation with better semantic meaning.
2)It has a clean nepali text function added which is used for better chunking for translation, this includes detecting devnagri symbols.
3) This model also includes sinhalese to english translation using mT5 model.
4)In this we have also upgrade the LitLingua pipeline to know the language choice from user and load the respective model. 
'''

"""
# =====================
# LitLingua OCR + Translation Pipeline (Nepali + Sinhalese)
# =====================
    This version supports:
   - Nepali → English translation using MarianMT
   - Sinhalese → English translation using mT5
   - Simple language choice input
   - OCR extraction with Tesseract
"""

import cv2
import pytesseract
from pdf2image import convert_from_path
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import textwrap
import torch

# ---------------------------
# CONFIG
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "nep": {
        "name": "Nepali",
        "ocr_lang": "nep",
        "model_type": "marian",
        "model_name": "iamTangsang/MarianMT-Nepali-to-English"
    },
    "sin": {
        "name": "Sinhalese",
        "ocr_lang": "sin",
        "model_type": "mt5",
        "model_name": "thilina/mt5-sinhalese-english"
    }
}


# ---------------------------
# STEP 1: OCR FUNCTIONS
# ---------------------------
def extract_text_from_image(image_path, lang="nep"):
    """Extract text from an image using Tesseract OCR"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang=lang)
    return text.strip()


def extract_text_from_pdf(pdf_path, lang="nep"):
    """Convert PDF pages to images and extract text"""
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPLER_PATH)
    full_text = ""
    for i, page in enumerate(pages):
        image_path = f"temp_page_{i}.png"
        page.save(image_path, "PNG")
        text = extract_text_from_image(image_path, lang=lang)
        full_text += text + "\n\n"
    return full_text.strip()


# ---------------------------
# STEP 2: TRANSLATION
# ---------------------------
def load_translation_model(lang_key):
    cfg = MODELS[lang_key]
    print(f"Loading {cfg['name']} → English model...")
    if cfg["model_type"] == "marian":
        tokenizer = MarianTokenizer.from_pretrained(cfg["model_name"])
        model = MarianMTModel.from_pretrained(cfg["model_name"]).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"]).to(device)
    print("✅ Model loaded successfully")
    return tokenizer, model


def translate_text(text, tokenizer, model, chunk_size=400):
    """Translate long text by splitting into chunks"""
    sentences = textwrap.wrap(text, chunk_size)
    translated_output = []
    for i, chunk in enumerate(sentences):
        print(f"🔹 Translating chunk {i + 1}/{len(sentences)}...")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(device)
        translated = model.generate(**inputs, max_length=512)
        english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_output.append(english_text)
    return "\n".join(translated_output)


# ---------------------------
# STEP 3: PIPELINE
# ---------------------------
def litlingua_pipeline(pdf_path, lang_key):
    cfg = MODELS[lang_key]
    print(f"\n📘 Extracting {cfg['name']} text from PDF...")
    native_text = extract_text_from_pdf(pdf_path, lang=cfg["ocr_lang"])

    print("🌐 Loading translation model...")
    tokenizer, model = load_translation_model(lang_key)

    print("🔁 Translating to English...")
    english_text = translate_text(native_text, tokenizer, model)

    print("\n✅ Translation Complete!\n")
    print("--------Original Text--------\n")
    print(native_text[:500], "...\n")
    print("--------Translated Text--------\n")
    print(english_text[:500], "...\n")

    with open("translated_output.txt", "w", encoding="utf-8") as f:
        f.write(english_text)
    print("📝 Saved translation to 'translated_output.txt'")


# ---------------------------
# STEP 4: RUN
# ---------------------------
if __name__ == "__main__":
    print("Select source language:")
    print("1) Nepali 🇳🇵")
    print("2) Sinhalese 🇱🇰")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        lang_key = "nep"
    elif choice == "2":
        lang_key = "sin"
    else:
        print("Invalid input! Defaulting to Nepali.")
        lang_key = "nep"

    pdf_path = "sample_nepali2.pdf"  # Change your input file here
    litlingua_pipeline(pdf_path, lang_key)

    

