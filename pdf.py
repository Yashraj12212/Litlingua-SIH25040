from fpdf import FPDF

# Install fpdf if not already
# pip install fpdf

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=16)

# Sample Nepali text
lines = [
    "नेपाल एक सुन्दर देश हो।",
    "सगरमाथा संसारको सबैभन्दा अग्लो पर्वत हो।",
    "हामीलाई हाम्रो संस्कृतिको गर्व छ।",
    "पाठशालामा नेपाली भाषा पढाइन्छ।"
]

for line in lines:
    pdf.cell(200, 10, txt=line, ln=True, align='L')

# Save PDF
pdf.output("sample_nepali.pdf")

print("sample_nepali.pdf created successfully!")
