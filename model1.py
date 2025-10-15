#this is the first and most rough version of the code, only for understand how the model works


# Image and PDF processing
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Translation
from transformers import MarianMTModel, MarianTokenizer


# Replace this path with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#ocr funtion to get the text from image
def extract_text_from_image(image_path, lang='nep'):

    # Extract text from a single image using Tesseract OCR
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    text = pytesseract.image_to_string(thresh, lang=lang)
    return text

#converting pdf to images then using the ocr funtion
def extract_text_from_pdf(pdf_path, lang='nep'):
    """
    Convert PDF pages to images and extract text
    """
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    
    for i, page in enumerate(pages):
        image_path = f"temp_page_{i}.png"
        page.save(image_path, 'PNG')
        text = extract_text_from_image(image_path, lang=lang)
        full_text += text + "\n"
    
    return full_text

# Load translation model
model_name = "iamTangsang/MarianMT-Nepali-to-English"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

#pipeline 
# Example usage (replace with your PDF/image)

pdf_path = "sample_nepali2.pdf"
native_text = extract_text_from_pdf(pdf_path, lang='nep')
english_text = translate_text(native_text)

print("--------Original Text--------")
print(native_text[:500])  # show first 500 chars
print("\n--------Translated Text--------")
print(english_text[:500])


