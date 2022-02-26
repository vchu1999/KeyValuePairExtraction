import utils
import os
import glob

for a in range(1):
    utils.generate_documents_continuous()

# utils.csv_parse()
# text_file_name = r'document_status.txt'
# status_file_object = open(text_file_name, 'r')
# a = status_file_object.readline()
# print(a)
# print(a=='0')
# status_file_object.close()
#
# status_file_object = open(text_file_name, 'w+')
# status_file_object.write("3")
# status_file_object.close()

# folder_path = 'generated_dictionaries'
# for filename in glob.glob(os.path.join(folder_path, '*.txt')):
#     with open(filename, 'r') as f:
#         text = f.read()
#         print(text.split('/////'))
#         print(filename)
#         print(len(text))