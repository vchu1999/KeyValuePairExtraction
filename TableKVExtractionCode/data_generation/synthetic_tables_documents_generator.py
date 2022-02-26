from shutil import copyfile
from PIL import Image
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import random


def generate_documents_continuous(num_documents=100, random_diagrams=False, min_diagrams=2, max_diagrams=10):
    """
    Generates synthetic data set for documents and stores the ground truth key-value pairs for them on disk
    Random diagrams can be added for extra noise
    Pyplot has issue with garbage collection, so only 368 documents can be generated at a time
    This code has to be run multiple times if more than 368 documents are needed
    By default 100 are generated at a time
    :param num_documents: number of documents to generate in one run, default is 100
    :param random_diagrams: boolean if random diagrams are desired, default is false
    :param min_diagrams: minimum number of diagrams if random diagrams are desired
    :param max_diagrams: maximum number of diagrams if random diagrams are desired
    :return: list of dictionaries of the document data
    """
    source_folder_name = 'document_generation_source'  # the name of the folder with pre-generated diagrams
    diagram_name = 'diagram_'  # prefix of all diagram files
    num_source_diagrams = 5  # number of diagrams
    white_page_name = 'white_page.jpg'  # name of image file of a white page
    destination_folder_name = '../generated_documents'  # stores generated documents
    dictionary_folder_name = '../generated_dictionaries'  # stores ground truth key-value pairs

    # retrieve the index of the next document to be generated
    status_file_name = r'document_status.txt'
    status_file_object = open(status_file_name, 'r')
    next_index = status_file_object.readline()
    next_index = int(next_index)
    status_file_object.close()

    if random_diagrams:  # opens all diagrams and saves them if random diagrams are desired
        diagrams = []
        diagram_dimensions = []
        for index_diagram in range(num_source_diagrams):
            file_diagram_index = index_diagram + 1
            diagrams.append(Image.open(source_folder_name + '/' + diagram_name + str(file_diagram_index) + '.jpg'))
            diagram_dimensions.append(diagrams[index_diagram].size)

    white_image = Image.open(source_folder_name + '/' + white_page_name)
    base_size = white_image.size

    all_documents_data = []

    for index in range(num_documents):
        actual_index = next_index + index
        document_data = {}
        new_file_name = destination_folder_name + '/' + f'{actual_index:10d}' + '.jpg'
        document_data['document_image_path'] = new_file_name
        copyfile(source_folder_name + '/' + white_page_name, new_file_name)
        new_document = Image.open(new_file_name)

        # store boundaries of objects already placed on the document in (x1, y1, x2, y2) format.
        object_locations = []

        kv_pairs = create_random_kv_pairs()
        create_table_image(kv_pairs)

        new_table = Image.open('created_table.png')
        new_table_width, new_table_height = new_table.size
        new_coord_x, new_coord_y = find_rand_coord(base_size, [new_table_width, new_table_height], object_locations)
        object_locations.append(
            (new_coord_x, new_coord_y, new_coord_x + new_table_width, new_coord_y + new_table_height))
        new_document.paste(new_table, [new_coord_x, new_coord_y])

        # adds random diagrams if desired
        if random_diagrams:
            num_diagrams = randint(min_diagrams, max_diagrams)
            for iteration in range(num_diagrams):
                random_diagram_index = randint(0, num_source_diagrams - 1)
                diagram = diagrams[random_diagram_index]
                diagram_size = diagram_dimensions[random_diagram_index]
                new_coord = find_rand_coord(base_size, diagram_size, object_locations)

                # adds the new diagram if possible
                if new_coord is not None:
                    object_locations.append(
                        (new_coord[0], new_coord[1], new_coord[0] + diagram_size[0], new_coord[1] + diagram_size[1]))
                    new_document.paste(diagram, new_coord)

        new_document.save(new_file_name)
        document_data.update(kv_pairs)
        all_documents_data.append(document_data)

    # store next file number in document_status for when running the script multiple times
    status_file_object = open(status_file_name, 'w+')
    status_file_object.write(str(next_index + num_documents))
    status_file_object.close()

    dictionary_in_string = ''
    for document_data in all_documents_data:
        dictionary_string = ''
        for key in document_data:
            dictionary_string = dictionary_string + ':::' + key + ':::' + document_data[key]
        if len(dictionary_string) > 0:
            dictionary_string = dictionary_string[3:]  # gets rid of extra ':::' at beginning
            dictionary_in_string = dictionary_in_string + '/////' + dictionary_string
    dictionary_in_string = dictionary_in_string[5:]  # gets rid of extra '/////' at beginning

    dictionary_file_name = dictionary_folder_name + '/' + str(next_index) + '-' + str(
        next_index + num_documents - 1) + '.txt'
    dictionary_file = open(dictionary_file_name, 'w')
    dictionary_file.write(dictionary_in_string)
    dictionary_file.close()

    return all_documents_data


# can be redone to be faster deterministically
def find_rand_coord(base_size, object_size, object_locations, num_tries=100):
    """
    Finds a random coordinate where an object can be inserted into the document.
    :param base_size: size of the base (width, height)
    :param object_size: size of the object to be placed on base (width, height)
    :param object_locations: list of locations of other objects on the base (x1, y1, x2, y2)
    :param num_tries: number of tries for finding a possible coordinate, default is 100
    :return: x and y coordinates that the object can be placed at
    """
    x_max = base_size[0] - object_size[0]
    y_max = base_size[1] - object_size[1]

    x_guess = None
    y_guess = None

    for attempt in range(num_tries):
        x_guess = randint(0, x_max)
        y_guess = randint(0, y_max)

        satisfied = True
        for object_location in object_locations:
            if diagrams_overlap((x_guess, y_guess), (x_guess + object_size[0], y_guess + object_size[1]),
                                (object_location[0], object_location[1]), (object_location[2], object_location[3])):
                satisfied = False
                x_guess = None
                y_guess = None
                break
        if satisfied:
            break
    if x_guess is None:
        return None
    else:
        return x_guess, y_guess


def diagrams_overlap(ulc1, brc1, ulc2, brc2):
    """
    Checks if two objects would overlap in a document
    :param ulc1: upper left corner of first diagram
    :param brc1: bottom right corner of first diagram
    :param ulc2: upper left corner of second diagram
    :param brc2: bottom right corner of second diagram
    :return: boolean if the two diagrams overlap
    """
    # checks if they horizontally overlap each other
    if ulc1[0] >= brc2[0] or ulc2[0] >= brc1[0]:
        return False

    # checks if they vertically overlap each other
    if brc1[1] <= ulc2[1] or brc2[1] <= ulc1[1]:
        return False

    return True


def create_table_image(kv_pairs):
    """
    generates a randomly generated table image
    table saved as "created_table.png"
    :param kv_pairs: dictionary of key:value pairs
    :return: void
    """
    # table formats:
    # 1: k v/ k v/ k v/ ... (horizontal orientation)
    # 2: k k k ... / v v v ... (vertical orientation)
    table_format = randint(1, 2)
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = None

    if table_format == 1:  # horizontal orientation
        first_column_key = None
        first_column = []
        second_column_key = None
        second_column = []
        for key in kv_pairs.keys():
            if first_column_key is None:
                first_column_key = key
                second_column_key = kv_pairs[key]
            else:
                first_column.append(key)
                second_column.append(kv_pairs[key])

        # adds empty cells to bottom of table if only 1 key-value pair provided
        if len(first_column) == 0:
            first_column.append(' ')
            second_column.append(' ')

        df = pd.DataFrame({first_column_key: first_column, second_column_key: second_column})

    elif table_format == 2:  # vertical orientation
        new_input_dict = {}
        for key in kv_pairs.keys():
            new_input_dict[key] = [kv_pairs[key]]
        df = pd.DataFrame(new_input_dict)

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()

    plt.savefig('created_table.png')

    plt.clf()
    plt.close()


def create_random_kv_pairs():
    """
    Randomly generates key-value pairs
    :return: dictionary of key:value pairs
    """
    kv_pairs = {}
    sample_dictionary = {'color': ['red', 'blue', 'green', 'purple'],
                         'height': ['tall', 'short'],
                         'temperature': ['hot', 'cold', 'warm', 'cool'],
                         'shape': ['circle', 'square', 'triangle', 'oval'],
                         'size': ['big', 'small'],
                         'taste': ['sweet', 'sour', 'spicy', 'salty']}

    # Other parameters can be added such as the following below

    # sample_dictionary['product number'] = ['asdfe2rer4', 'sdf-34ds', 'x345-er', 'p45g', '34dg']
    # sample_dictionary['length'] = ['1', '2', '3', '4', '5']
    # sample_dictionary['max temperature'] = ['60', '65', '70', '75', '100']
    # sample_dictionary['min temperature'] = ['10', '25', '40', '50']

    while len(kv_pairs) == 0:  # keep going until there is at least one key-value pair
        for key in sample_dictionary.keys():
            if random.random() < 0.5:  # each parameter has a 50% chance of being included
                random_index = randint(0, len(sample_dictionary[key]) - 1)
                kv_pairs[key] = sample_dictionary[key][random_index]
    return kv_pairs


def main():
    generate_documents_continuous()


if __name__ == "__main__":
    main()