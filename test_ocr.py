import pytesseract
from PIL import Image

# Update this path if your Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

text = pytesseract.image_to_string(Image.open("sample_nepali1.png"), lang="nep")
print("Extracted text:", text)
