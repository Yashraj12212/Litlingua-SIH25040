#this is the first working version of the code, more optimized and better version is in model2.py,
# this code does the translation but the output is semantically wrong

# =====================
# LitLingua OCR + Translation Pipeline
# =====================

import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
import textwrap

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------------------
# STEP 1: OCR FUNCTIONS
# ---------------------------

def extract_text_from_image(image_path, lang='nep'):
    """Extract Nepali text from a single image using Tesseract OCR"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang=lang)
    return text.strip()


def extract_text_from_pdf(pdf_path, lang='nep'):
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=r"C:\poppler-24.08.0\Library\bin")
    full_text = ""
    for i, page in enumerate(pages):
        image_path = f"temp_page_{i}.png"
        page.save(image_path, 'PNG')
        text = extract_text_from_image(image_path, lang=lang)
        full_text += text + "\n\n"
    return full_text.strip()



# ---------------------------
# STEP 2: TRANSLATION MODEL
# ---------------------------

print("Loading translation model... (This may take a minute)")
model_name = "iamTangsang/MarianMT-Nepali-to-English"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
print("Model loaded successfully ✅")


def translate_text(text, chunk_size=400):
    """Translate large Nepali text to English in smaller chunks"""
    sentences = textwrap.wrap(text, chunk_size)
    translated_output = []
    for chunk in sentences:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        translated = model.generate(**inputs)
        english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_output.append(english_text)
    return "\n".join(translated_output)


# ---------------------------
# STEP 3: PIPELINE
# ---------------------------

def litlingua_pipeline(pdf_path):
    print("Extracting Nepali text from PDF...")
    native_text = extract_text_from_pdf(pdf_path)
    print("Translating to English...")
    english_text = translate_text(native_text)
    
    print("\n✅ Translation Complete!")
    print("\n--------Original (Nepali)--------\n")
    print(native_text[:500], "...")
    print("\n--------Translated (English)--------\n")
    print(english_text[:500], "...")
    
    # Optional: Save to file
    with open("translated_output.txt", "w", encoding="utf-8") as f:
        f.write(english_text)
    print("\n📝 Saved translation to 'translated_output.txt'")


# ---------------------------
# STEP 4: RUN
# ---------------------------

if __name__ == "__main__":
    pdf_path = "sample_nepali2.pdf"   # Replace with your file
    litlingua_pipeline(pdf_path)
