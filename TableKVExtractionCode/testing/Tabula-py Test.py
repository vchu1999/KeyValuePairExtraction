import tabula
import pytesseract
from TableKVExtractionCode import utils

# df = tabula.read_pdf(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\sl9126n-osra.pdf", pages='all')
# df = tabula.read_pdf(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\887downloader.phppsd-pmaxxx-mmtbl-cxx-xft.pdf", pages='all')
#
# print(df)
# images = utils.pdf_to_image(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\arrowtest0.jpg")
# print(pytesseract.image_to_data(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\arrowtest0.jpg"))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# print(pytesseract.image_to_data(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\arrowtest0.jpg"))
 

images = utils.pdf_to_image(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\pdfresizer.com-pdf-crop.pdf")
pdf = pytesseract.image_to_pdf_or_hocr(images[0], extension='pdf')
with open('../sample_documents/test.pdf', 'w+b') as f:
    f.write(pdf)  # pdf type is bytes by default
df = tabula.read_pdf('test.pdf', pages='all')

print(df)