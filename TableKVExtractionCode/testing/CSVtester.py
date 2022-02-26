from TableKVExtractionCode import utils

# urlretrieve('http://download.siliconexpert.com/pdfs/2012/2/15/7/2/58/796/mrtnl_/manual/243172782482997c24e1cb.pdf', 'path_to_save_plus_some_file.pdf')
paths = [r"C:\Users\vchu1\Downloads\OCR_Sample_V1_SiliconExpert_confidential.xlsx - Connector Headers and PCB Recep.csv"]
utils.csv_parse(paths)