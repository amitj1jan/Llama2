from PyPDF2 import PdfReader
from io import BytesIO
import chainlit as cl
# import docx2txt
# import docx
import os

encoding = 'utf-8'

def pdf_to_text(file):
    # Read the PDF file
    text_stream = BytesIO(file.content)
    pdf = PdfReader(text_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    
    return(pdf_text)

# def doc_to_text(file):
#     # Read the PDF file
# #     text_stream = BytesIO(file.content)
#     text = docx2txt.process(file)
    
#     print(text)
#     text = text.decode(encoding)    
#     return(text)


def doc_to_text(file):
    doc = docx.Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def write_text(path, text):
    with open(os.path.join(path, 'text.txt'), 'w') as fp:
        fp.write(text)
    
