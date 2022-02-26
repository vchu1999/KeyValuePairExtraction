import pytesseract
import cv2


'''
Finds key and value from an image, assuming it has both.
'''


def self_contained_extract(img):
    # Replace this with path to your own tesseract.exe installation
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confidences = data['conf']
    texts = data['text']
    heights = data['height']

    # for debugging
    print(confidences)
    print(texts)
    print(heights)

    words = []
    word_heights = []
    # find the valid words and their heights
    for index, confidence in enumerate(confidences):
        if confidence != '-1' and not texts[index].isspace() and texts[index] != '':
            words.append(texts[index])
            word_heights.append(heights[index])

    # Attempt 1: Check if key and value text sizes vary
    biggest_diff = 0
    biggest_diff_index = 0
    for index in range(len(words) - 1):
        if biggest_diff < abs(word_heights[index + 1] - word_heights[index]):
            biggest_diff = abs(word_heights[index + 1] - word_heights[index])
            biggest_diff_index = index
    average = sum(word_heights) / len(word_heights)

    # Attempt 1: Check if key and value text sizes vary

    # if biggest jump/drop in height is substantial compared to average of height,
    # split key/value using height as the metric
    if biggest_diff > average * 0.1:
        key_list = words[:biggest_diff_index + 1]
        value_list = words[biggest_diff_index + 1:]
        if len(key_list) != 0 and len(value_list) != 0:
            key = ' '.join(key_list)
            value = ' '.join(value_list)
            print('key is: ' + key)
            print('value is: ' + value)
            return [key, value]

    # Attempt 2: split key and value based on the largest space.

    current_space_start_index = -1
    current_space_length = 0
    longest_space_start_index = -1
    longest_space_length = 0
    first_word_encountered = False

    for index, confidence in enumerate(confidences):
        if confidence != '-1' and not texts[index].isspace() and texts[index] != '':
            first_word_encountered = True
            if current_space_length > longest_space_length:
                longest_space_start_index = current_space_start_index
                longest_space_length = current_space_length
            current_space_start_index = -1
            current_space_length = 0
        else:
            if first_word_encountered:
                if current_space_length == 0:
                    current_space_start_index = index
                    current_space_length += 1
                else:
                    current_space_length += 1

    if longest_space_start_index > -1:
        key_list = []
        value_list = []
        for index, confidence in enumerate(confidences):
            if confidence != '-1' and not texts[index].isspace() and texts[index] != '':
                if index < longest_space_start_index:
                    key_list.append(texts[index])
                else:
                    value_list.append(texts[index])
        key = ' '.join(key_list)
        value = ' '.join(value_list)
        print('key is: ' + key)
        print('value is: ' + value)
        return [key, value]
    else:
        print('key and value cannot be determined')
        return []

# Enter the path to image of cell
img = cv2.imread(r'C:\Users\vchu1\OneDrive\Documents\cells\15.PNG')
self_contained_extract(img)