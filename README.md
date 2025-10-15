# LitLingua – SIH25040

## Project Overview
LitLingua is an AI-powered OCR and translation pipeline that converts printed Nepali and Sinhalese literature into English.  
It aims to bridge the language gap and preserve South Asian literary heritage by making regional texts accessible to a global audience.

---

## Features
- OCR Extraction: Recognizes Nepali and Sinhalese text from scanned books, PDFs, and images.  
- AI Translation: Uses multilingual transformer models (like NLLB-200) for accurate translation to English.  
- Text Cleaning & Preprocessing: Automatically denoises, enhances, and prepares text for better OCR accuracy.  
- Multi-format Input Support: Works with both images and PDFs.  
- Simple CLI Interface: Easy-to-run pipeline for non-technical users.  
- Future Ready: Supports self-learning lexicon for improved translations over time.

---

## Tech Stack

| Layer | Tools / Libraries |
|-------|--------------------|
| OCR | Tesseract OCR, PaddleOCR, OpenCV |
| Translation | Facebook NLLB-200, MarianMT, HuggingFace Transformers |
| Preprocessing | pdf2image, Pillow, OpenCV |
| Backend | Python |
| Output | FPDF, ReportLab (for translated text export) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Yashraj12212/Litlingua-SIH25040.git
cd Litlingua-SIH25040

# Create a virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate        # On Windows
# or
source venv/bin/activate     # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

```
## Usage

```

# Step 1: Run the main pipeline
python main.py

# Step 2: Select your source language
# (1) Nepali
# (2) Sinhalese

# Step 3: Provide input file path (PDF/Image)
# Example:
# data/sample_nepali.pdf

# Step 4: Get translated English text in output/

```
## Project Structure

```

Litlingua-SIH25040/
│
├── data/                 # Input PDFs or images
├── output/               # Translated text or reports
├── src/
│   ├── ocr_module.py     # Handles Tesseract OCR
│   ├── translate_module.py # Handles NLLB/MarianMT translation
│   ├── preprocess.py     # Image cleanup & PDF conversion
│   └── utils.py          # Helper functions
│
├── requirements.txt      # Dependencies
├── main.py               # Entry point for CLI
└── README.md             # Project documentation

```
## Example Output

```

Input:  Nepali scanned PDF (page_1.png)
↓
OCR → Clean text → Translation (NLLB-200)
↓
Output: English text file saved at /output/translated_page_1.txt

```
## Future Scope


```

Add context-aware fine-tuning for literary tone retention.

Develop a mobile app for real-time OCR and translation.

Implement offline mode using ONNX or TensorFlow Lite.

```
## Tagline

```

✨ "Preserving South Asian literature — one page at a time."