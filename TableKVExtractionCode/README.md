# arrow-electronics-data-extraction
Key Value extraction for Arrow electronics technical specs - CSAIL UROP 

Needs to install pytesseract and download glove.6B.50d.txt in the root folder.

To run, go to folder in terminal and run:

python main.py path_to_pdf

or the following if you want to see intermediate steps visualized:

python main.py path_to_pdf -d

If tables are not detected correctly, change the thickness and threshold variables in get_horizontal_lines and get_vertical_lines in table_cell_extraction.py.

To generate synthetic dataset, run data_generation/synthetic_tables_documents_generator.py

To train the model, run ranking_model_pipeline_multithreading.py
