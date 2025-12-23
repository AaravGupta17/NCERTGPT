import pdfplumber

with pdfplumber.open("tests/test.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text())
