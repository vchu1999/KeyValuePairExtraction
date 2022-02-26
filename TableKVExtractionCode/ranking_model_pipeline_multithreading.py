import cv2
import torch
from TableKVExtractionCode import utils, ranking_model_more
from TableKVExtractionCode import vectorization
from TableKVExtractionCode import table_cell_extraction
import numpy as np
import torch.nn as nn
import torch.optim as optim
import datetime
import concurrent.futures
import math


def train(paths=None):
    """
    Trains and saves a model

    :param path: location of csv file with all the required data
    :return: none
    """
    # Load the GloVe dataset
    embeddings_dict = {}
    glove_iter = 0
    with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        # iterate through each line
        for i, line in enumerate(f):
            # split the line at every space
            values = line.split()
            # each word appears first
            word = values[0]
            # the numbers after the word are its vector
            vector = np.asarray(values[1:], "float32")
            # add the word and its vector to the dictionary
            embeddings_dict[word] = vector
            glove_iter += 1
            if glove_iter % 10000 == 0:
                print("Processed GloVe lines: " + str(glove_iter))

    # average all vectors from GloVe dataset
    average_vec = np.mean(list(embeddings_dict.values()), axis=0)

    # define number of epochs to run
    num_epochs = 1000

    # parse csv file
    if paths is None:
        data = utils.csv_parse()
    else:
        data = utils.csv_parse(paths)
    # data = [data[0]]

    param_list = []
    data_limit = 10000000
    read_count = 0
    test_data_start = 90
    param_list_test = []
    for index, document in enumerate(data):
        if index < test_data_start:
            param_list.append(ocr_document_param(document, paths, embeddings_dict, average_vec))
        else:
            param_list_test.append(ocr_document_param(document, paths, embeddings_dict, average_vec))
        read_count = read_count + 1

        if read_count >= data_limit:
            break
    start_multi = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(ocr_document, param) for param in param_list]

    results = [f.result() for f in futures]
    parsed_data = [item for sublist in results for item in sublist]
    end_multi = datetime.datetime.now()
    print(end_multi - start_multi)
    print(parsed_data)
    print(len(parsed_data))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures_test = [executor.submit(ocr_document, param) for param in param_list_test]

    results_test = [f.result() for f in futures_test]
    parsed_data_test = [item for sublist in results_test for item in sublist]
    print("Training data size: " + str(len(parsed_data)))
    print("Testing data size: " + str(len(parsed_data_test)))

    ranking_model = ranking_model_more.ranking_model(54, 120)
    # previously 120
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(ranking_model.parameters(), lr=0.001)

    epoch_timestamp = datetime.datetime.now()

    best_accuracy = 0
    best_accuracy_epoch = -1

    best_test_accuracy = 0
    best_test_accuracy_epoch = -1

    for epoch in range(num_epochs):
        epoch_loss = 0
        ranking_model.train()

        for data in parsed_data:
            vectors = data[0]
            association = data[1]
            vectors_tensor = torch.from_numpy(np.hstack(vectors))

            for index, vector in enumerate(vectors):
                association_answer = association[index].unsqueeze(0).long()
                if association_answer >= 0:
                    vector_tensor = torch.from_numpy(vector)
                    output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                    loss = loss_function(torch.unsqueeze(output, 0), association_answer)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    epoch_loss = epoch_loss + loss
        previous_time = epoch_timestamp
        epoch_timestamp = datetime.datetime.now()

        print("Epoch: " + str(epoch))
        print("Loss: " + str(epoch_loss))
        print("Epoch time: " + str(epoch_timestamp - previous_time) + ' ms')
        if epoch % 1 == 0:
            accuracy = evaluate(ranking_model, parsed_data)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
            print("Best accuracy was: " + str(best_accuracy) + " at epoch " + str(best_accuracy_epoch))
        if epoch % 1 == 0:
            accuracy = evaluate(ranking_model, parsed_data_test)
            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                best_test_accuracy_epoch = epoch
            print("Best test accuracy was: " + str(best_test_accuracy) + " at epoch " + str(best_test_accuracy_epoch))

    evaluate(ranking_model, parsed_data)


def ocr_document(ocr_document_params):
    """
    Applies OCR to a table in a document

    :param ocr_document_params: set of parameters including a document and path
    :return: returns parsed_data, a collection of key-value pairs and strings
    """
    document, paths, embeddings_dict, average_vec = ocr_document_params.get_values()
    image_path = document['document_image_path']
    parsed_image_paths = []
    parsed_data = []
    if paths is not None:
        parsed_image_paths = utils.pdf_to_image(image_path)
        print(parsed_image_paths)
    else:
        parsed_image_paths = [image_path]

    for parsed_image_path in parsed_image_paths:
        img = cv2.imread(parsed_image_path)
        horiz = table_cell_extraction.get_horizontal_lines(img)
        vert = table_cell_extraction.get_vertical_lines(img)
        lines = horiz + vert

        out, boxes = table_cell_extraction.get_cells(lines)
        tables, neighbors = table_cell_extraction.get_tables(boxes)

        for table in tables:

            if len(table) > 1:
                print(len(table))
                boundary = table_cell_extraction.get_table_boundary(table)
                cropped = img[boundary[1]: boundary[1] + boundary[3], boundary[0]: boundary[0] + boundary[2]]
                print(table)

                if paths is None:
                    vectors, association, kv, parsed_strings, right, down = vectorization.table_to_vectors(table,
                                                                                                           neighbors,
                                                                                                           image_path,
                                                                                                           document,
                                                                                                           embeddings_dict,
                                                                                                           average_vec,
                                                                                                           pre_jpg=True)
                else:
                    vectors, association, kv, parsed_strings, right, down = vectorization.table_to_vectors(table,
                                                                                                           neighbors,
                                                                                                           image_path,
                                                                                                           document,
                                                                                                           embeddings_dict,
                                                                                                           average_vec,
                                                                                                           pre_jpg=False)
                data_point = [vectors, association, kv, parsed_strings, right, down]
                parsed_data.append(data_point)
    return parsed_data


def evaluate(ranking_model, parsed_data):
    """
    Evaluates the accuracy of a model

    :param ranking_model: the model to evaluate
    :param parsed_data: the inputs and expected outputs
    :return: fractional accuracy of model
    """
    # specifies which method of finding accuracy to use
    greedy = False
    if greedy:
        return evaluate_greedy(ranking_model, parsed_data)
    else:
        return evaluate_exhaustive(ranking_model, parsed_data)


def evaluate_exhaustive(ranking_model, parsed_data):
    """
    Evaluates the accuracy of a model using an exhaustive algorithm

    :param ranking_model: the model to evaluate
    :param parsed_data: the inputs and expected outputs
    :return: fractional accuracy of model
    """
    # switch the model to evaluation mode
    ranking_model.eval()
    correct_identified = 0
    incorrect_identified = 0
    num_documents = 0
    num_actual_documents = 0
    for data in parsed_data:
        num_documents += 1
        vectors = data[0]
        association = data[1]
        parsed_strings = data[3]
        right = data[4]
        down = data[5]

        vectors_tensor = torch.from_numpy(np.hstack(vectors))
        right_score = 0
        down_score = 0
        counted_actual = False
        for index, vector in enumerate(vectors):
            if association[index].unsqueeze(0).long() >= 0 and \
                    [association[index].unsqueeze(0).long()] is not None:
                if not counted_actual:
                    num_actual_documents += 1
                    counted_actual = True
                vector_tensor = torch.from_numpy(vector)
                output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                output[index] = 0
                index_np = 0
                index_np_array = np.empty_like(parsed_strings, dtype=bool)
                for a in parsed_strings:
                    if a is None:
                        index_np_array[index_np] = True
                    else:
                        index_np_array[index_np] = False
                    index_np = index_np + 1
                output[torch.from_numpy(index_np_array) == True] = -math.inf

                right_index = right[index]
                down_index = down[index]
                right_score = right_score + output[right_index]
                down_score = down_score + output[down_index]

        if right_score > down_score:
            for index, vector in enumerate(vectors):
                if association[index].unsqueeze(0).long() >= 0 and \
                        parsed_strings[association[index].unsqueeze(0).long()] is not None:
                    vector_tensor = torch.from_numpy(vector)
                    output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                    output[index] = 0
                    index_np = 0
                    index_np_array = np.empty_like(parsed_strings, dtype=bool)
                    for a in parsed_strings:
                        if a is None:
                            index_np_array[index_np] = True
                        else:
                            index_np_array[index_np] = False
                        index_np = index_np + 1
                    output[torch.from_numpy(index_np_array) == True] = -math.inf

                    right_index = right[index]
                    if right_index == association[index].unsqueeze(0).long():
                        correct_identified += 1
                    else:
                        incorrect_identified += 1
        else:
            for index, vector in enumerate(vectors):
                if association[index].unsqueeze(0).long() >= 0 \
                        and parsed_strings[association[index].unsqueeze(0).long()] is not None:
                    vector_tensor = torch.from_numpy(vector)
                    output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                    output[index] = 0
                    index_np = 0
                    index_np_array = np.empty_like(parsed_strings, dtype=bool)
                    for a in parsed_strings:
                        if a is None:
                            index_np_array[index_np] = True
                        else:
                            index_np_array[index_np] = False
                        index_np = index_np + 1
                    output[torch.from_numpy(index_np_array) == True] = -math.inf

                    down_index = down[index]
                    if down_index == association[index].unsqueeze(0).long():
                        correct_identified += 1
                    else:
                        incorrect_identified += 1

    print('correct: ' + str(correct_identified))
    print('incorrect: ' + str(incorrect_identified))
    print('accuracy: ' + str(correct_identified / (correct_identified + incorrect_identified)))
    print('total documents examined: ' + str(num_documents))
    print('total documents used: ' + str(num_actual_documents))
    return correct_identified / (correct_identified + incorrect_identified)


def evaluate_greedy(ranking_model, parsed_data):
    """
    Evaluates the accuracy of a model using a greedy algorithm

    :param ranking_model: the model to evaluate
    :param parsed_data: the inputs and expected outputs
    :return: fractional accuracy of model
    """
    # switch the model to evaluation mode
    ranking_model.eval()
    correct_identified = 0
    incorrect_identified = 0
    hit2 = 0
    hit3 = 0
    num_documents = 0
    num_actual_documents = 0
    for data in parsed_data:
        num_documents += 1
        vectors = data[0]
        association = data[1]
        parsed_strings = data[3]
        right = data[4]
        down = data[5]

        vectors_tensor = torch.from_numpy(np.hstack(vectors))
        counted_actual = False

        for index, vector in enumerate(vectors):
            if association[index].unsqueeze(0).long() >= 0 and \
                    parsed_strings[association[index].unsqueeze(0).long()] is not None:
                if not counted_actual:
                    num_actual_documents += 1
                    counted_actual = True

                vector_tensor = torch.from_numpy(vector)
                output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                output[index] = 0
                index_np = 0
                index_np_array = np.empty_like(parsed_strings, dtype=bool)
                for a in parsed_strings:
                    if a is None:
                        index_np_array[index_np] = True
                    else:
                        index_np_array[index_np] = False
                    index_np = index_np + 1
                output[torch.from_numpy(index_np_array) == True] = 0
                max_index = torch.argmax(output)
                values2, indices2 = torch.topk(output, 2)
                list_indices2 = indices2.tolist()
                values3, indices3 = torch.topk(output, 3)
                list_indices3 = indices3.tolist()

                # print(output.tolist())
                # print(max_index.item())
                if max_index.item() == association[index].unsqueeze(0).long():
                    correct_identified += 1
                else:
                    incorrect_identified += 1
                if association[index].unsqueeze(0).long() in list_indices2:
                    hit2 += 1
                if association[index].unsqueeze(0).long() in list_indices3:
                    hit3 += 1

    print('correct: ' + str(correct_identified))
    print('incorrect: ' + str(incorrect_identified))
    print('hit2: ' + str(hit2))
    print('hit3: ' + str(hit3))
    print('accuracy: ' + str(correct_identified / (correct_identified + incorrect_identified)))
    print('hit2 ratio: ' + str(hit2 / (correct_identified + incorrect_identified)))
    print('hit3 ratio: ' + str(hit3 / (correct_identified + incorrect_identified)))
    print('total documents examined: ' + str(num_documents))
    print('total documents used: ' + str(num_actual_documents))
    return correct_identified / (correct_identified + incorrect_identified)


class ocr_document_param:
    """
    A class of objects to store parameters so OCR can be applied to a document
    """
    def __init__(self, document, path, embeddings_dict, average_vec):
        self.document = document
        self.path = path
        self.embeddings_dict = embeddings_dict
        self.average_vec = average_vec

    def get_values(self):
        return self.document, self.path, self.embeddings_dict, self.average_vec

train()
