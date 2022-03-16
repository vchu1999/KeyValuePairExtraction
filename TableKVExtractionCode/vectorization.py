import cv2
import numpy as np
import pytesseract
import torch
import Levenshtein
import utils


def table_to_vectors(table, neighbors, image_path, document, embeddings_dict, unknown_vec, pre_jpg=False):
    """
    Encode cells in a table into vectors
    all outputs except vectors are lists with the same length of table input where each element represents a cell
    In associations, the value at each index is the index of the cell that is associated with
                     the cell at the index in question, or -1 if there is no association
    In key_or_value, the value is 1 if its a key, 0 if its a value, and -1 if its neither
    In right, the value at each index is the index of the cell closest to the right of the cell at the index in question
    In down, the value at each index is the index of the cell closest downward to the cell at the index in question
    @param table: set of cells (x-coord of top left corner, y-coord of top left corner, width, height)
    @param neighbors: dictionary of neighbors for each cell (from table_cell_extraction.get_tables)
    @param image_path: path to image
    @param document: document image was obtained from
    @param embeddings_dict: dictionary that relates words to vectors
    @param unknown_vec: default vec if no good vectorization is found
    @param pre_jpg: True if image is already a jpg, default is false
    @return: (vectors, associations, key_or_value, parsed_strings, right, down)
    """
    vectors = []
    if pre_jpg:
        paths = [image_path]
    else:
        paths = utils.pdf_to_image(image_path)

    # TODO: Assumes the pdf has only one sheet
    image = cv2.imread(paths[0])

    key_or_value = np.empty(len(table), dtype=object)
    right = np.empty(len(table), dtype=object)
    down = np.empty(len(table), dtype=object)
    parsed_strings = np.empty(len(table), dtype=object)

    cells = []

    for index, cell in enumerate(table):
        cells.append(cell)

        # x_min, x_max, y_min, y_max in that order
        position_vector = np.array([cell[0], cell[0] + cell[2], cell[1], cell[1] + cell[3]])
        position_vector = np.reshape(position_vector, (4, 1))

        # None if no match. Otherwise outputs the string in either keyset or valueset that matches the cell's content.
        parsed_string = parse_string(image[cell[1]: cell[1] + cell[3], cell[0]: cell[0] + cell[2]], document)
        string_vector = string_embedding(parsed_string, embeddings_dict, unknown_vec)
        string_vector = np.reshape(string_vector, (len(string_vector), 1))

        # combine position vector and string vector for overall vector of that cell
        embedded_vector = np.concatenate((position_vector, string_vector), axis=0)
        vectors.append(embedded_vector)

        if parsed_string is not None:
            parsed_strings[index] = parsed_string

    for index, cell in enumerate(cells):
        neighbors_cell = neighbors[cell]
        right_cell = -1  # index of closest cell to the right of cell in question
        down_cell = -1  # index of closest cell below the cell in question
        for neighbor_cell in neighbors_cell:
            # if neighbor cell is to the right of cell in question
            if neighbor_cell[0] > cell[0] + 10:
                if right_cell > -1:
                    prev_right = cells[right_cell]
                    # checks if prev right cell is further from the cell in question than current neighbor vertically
                    if abs(prev_right[1] - cell[1]) > abs(neighbor_cell[1] - cell[1]):
                        right_cell = cells.index(neighbor_cell)
                else:
                    right_cell = cells.index(neighbor_cell)

            # if neighbor cell is below the cell in question
            if neighbor_cell[1] > cell[1] + 10:
                if down_cell > -1:
                    prev_down = cells[down_cell]
                    # checks if prev down cell is further from the cell in question than current neighbor horizontally
                    if abs(prev_down[0] - cell[0]) > abs(neighbor_cell[0] - cell[0]):
                        down_cell = cells.index(neighbor_cell)
                else:
                    down_cell = cells.index(neighbor_cell)

        right[index] = right_cell
        down[index] = down_cell

    # index of the associated cell in k-v pair. -1 if not associated with any cell.
    # initiate all associations to -1
    associations = torch.ones(len(parsed_strings))
    associations = torch.neg(associations)

    for index, cell_string in np.ndenumerate(parsed_strings):
        if cell_string is not None:
            if cell_string in document.keys():
                key_or_value[index] = 1
                pos = np.where(parsed_strings == document[cell_string])
                pos = pos[0]
                if len(pos) > 0:
                    associations[index] = pos[0]

        else:
            key_or_value[index] = -1

    return vectors, associations, key_or_value, parsed_strings, right, down


def parse_string(image, document, path_to_tesseract='path to pytesseract', threshold=0.5):
    """
    Extract string from image
    Turns the OCR output to nearest valid word in the input document
    If no sufficiently close words, it is disregarded since it is most likely not a word within interest
    Disregarded if string's Levenshtein distance is more than or equal to threshold * length of the string
    :param image: image with string
    :param document: document the image was extracted from
    :param path_to_tesseract: raw path to tesseract
    :param threshold: threshold to determine close words
    :return: word from image
    """
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confidences = data['conf']
    texts = data['text']

    # get all possible text from the image
    words = ''
    for index, confidence in enumerate(confidences):
        if confidence != '-1' and not texts[index].isspace():
            parsed_word = texts[index]
            if len(words) == 0:
                words = parsed_word
            else:
                words = words + ' ' + parsed_word

    closest_word = find_closest_word(words, document, threshold)

    return closest_word


def find_closest_word(word, document, threshold):
    """
    Finds the closest word in the document to the input word
    @param word: word in question
    @param document: document word was extracted from
    @param threshold: threshold that must be satisfied for word to be close enough
    @return: closest word or None if no word is close enough
    """
    shortest_distance = -1
    closest_word = ''

    # go through possible keys
    for key in document.keys():
        new_distance = Levenshtein.distance(key, word)
        if shortest_distance == -1 or shortest_distance > new_distance:
            shortest_distance = new_distance
            closest_word = key

    # go through possible values
    for value in document.values():
        new_distance = Levenshtein.distance(value, word)
        if shortest_distance == -1 or shortest_distance > new_distance:
            shortest_distance = new_distance
            closest_word = value

    if 0 <= shortest_distance < threshold * len(word):
        return closest_word
    else:
        return None


def string_embedding(text, embeddings_dict, unknown_vec):
    """
    Turn the strings into vectors based on embedding dictionary
    :param text: text to turn into vector
    :param embeddings_dict: dictionary that relates words to vectors
    :param unknown_vec: vector to return when no string embedding is found
    :return: vector representation of the text or unknown_vec
    """
    if text is None:
        return unknown_vec

    else:
        words = text.split()
        vectors = []

        # convert word to vector embedding
        for word in words:
            if word in embeddings_dict.keys():
                vectors.append(embeddings_dict[word])
            else:
                vectors.append(unknown_vec)

        # if there are vectors, return the average. otherwise return unknown_vec
        if len(vectors) == 0:
            return unknown_vec
        else:
            words_embedding = np.mean(vectors, axis=0)
            return words_embedding
