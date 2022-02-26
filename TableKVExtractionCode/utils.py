import datetime

from pdf2image import convert_from_path
import pytesseract
from urllib.request import urlretrieve
import csv
import warnings
import os
import glob


def pdf_to_image(path):
    """
    Converts pdfs to jpgs
    :param path: path to pdf
    :return: jpg image
    """
    images = convert_from_path(path)
    paths = []
    filename = path.replace(".pdf", "")
    for i in range(len(images)):
        file = filename + str(i) + '.jpg'
        images[i].save(file, 'JPEG')
        paths.append(file)
    return paths


def self_contained_check(img, threshold=1.3):
    """
    Determines whether a cell has both key and value in it (self-contained). Algorithmic approach.
    Cell considered to be self-contained if there is a significant difference in size between two texts
    Size based on the heights extracted from pytesseract
    :param img: Image of a cell
    :param threshold: threshold to decide whether two texts are significantly different in size, default is 1.3
    :return: boolean if the cell is determined to be self-contained
    """
    # extract image data with pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confidences = data['conf']
    texts = data['text']
    heights = data['height']

    smallest_size = -1
    largest_size = -1

    for index, confidence in enumerate(confidences):
        if confidence != '-1' and not texts[index].isspace():
            current_size = heights[index]

            # initialize smallest and largest sizes
            if smallest_size == -1:
                smallest_size = current_size
                largest_size = current_size

            # update sizes
            if current_size < smallest_size:
                smallest_size = current_size

            if current_size > largest_size:
                largest_size = current_size

            # check if there is significant difference
            if smallest_size * threshold < largest_size:
                return True

    return False

def csv_parse(paths=None, url_key_name='PRODUCT_EXTERNAL_DATASHEET', image_path_key='document_image_path', print_updates=False):
    """
    Stores the input true input key-value pairs in a list of dictionaries
    If paths=None, no input CSV file with ground truth and the pipeline is trained on synthetically generated documents
    :param paths: paths to the input CSV file and the documents, default is None
    :param url_key_name: key name of urls, in arrow dataset labelled "PRODUCT_EXTERNAL_DATASHEET"
    :param image_path_key: key name of image path
    :return: list of dictionaries of key-value pairs for each document
    """
    data = []

    # get data from an input CSV file
    if paths is not None:
        for path in paths:
            if print_updates:
                start_time = datetime.datetime.now()

            keys = []

            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    # keys are in first row
                    if line_count == 0:
                        for key in row:
                            keys.append(key)

                    # add values associated to each key
                    else:
                        values = {}
                        saved = False
                        for index, value in enumerate(row):
                            values[keys[index]] = value

                            # saves pdf
                            if keys[index] == url_key_name:
                                saved = True
                                urlretrieve(value, 'documents_folder/' + str(hash(value)) + '.pdf')
                                values[image_path_key] = 'documents_folder/' + str(hash(value)) + '.pdf'

                        if len(values) != len(keys) + 1:
                            warnings.warn("not all keys were used: " + str(values))
                        if not saved:
                            warnings.warn("document not saved: " + str(values))

                        data.append(values)
                    line_count += 1

                    if line_count % 10 == 0 and print_updates:
                        print("line_count processed in csv_parse: " + str(line_count))

                    if line_count > 20:
                        break

                if print_updates:
                    end_time = datetime.datetime.now()
                    print(line_count)
                    print(end_time - start_time)

    # Train on synthetic data if paths = None
    else:
        # generated using data_generation/synthetic_tables_documents_generator.py
        folder_path = 'generated_dictionaries'

        for filename in glob.glob(os.path.join(folder_path, '*.txt')):
            with open(filename, 'r') as f:
                synthetic_data = f.read()
                all_documents_data = synthetic_data.split('/////')

                for document_data in all_documents_data:
                    key_value_data = document_data.split(':::')
                    key_value_pairs = {}
                    key = None
                    for item in key_value_data:
                        if key is None:
                            key = item
                        else:
                            key_value_pairs[key] = item
                            key = None

                    data.append(key_value_pairs)

    return data

