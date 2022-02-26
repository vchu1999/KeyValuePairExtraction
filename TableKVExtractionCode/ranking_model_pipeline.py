import cv2
import torch
from TableKVExtractionCode import utils
from TableKVExtractionCode import table_cell_extraction
from TableKVExtractionCode import vectorization
import numpy as np
import torch.nn as nn
import torch.optim as optim
import TableKVExtractionCode.ranking_model_simple
import TableKVExtractionCode.ranking_model
import datetime



def train(paths=None):
    # Load the GloVe dataset
    embeddings_dict = {}
    glove_iter = 0
    with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
            glove_iter += 1
            if glove_iter % 10000 == 0:
                print("Processed GloVe lines: " + str(glove_iter))

    average_vec = np.mean(list(embeddings_dict.values()), axis=0)

    # take care of self_contained cells
    num_epochs = 40

    if paths is None:
        data = utils.csv_parse()
    else:
        data = utils.csv_parse(paths)
    # data = [data[0]]

    parsed_data = []

    for document in data:
        image_path = document['document_image_path']
        parsed_image_paths = []
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
            tables = table_cell_extraction.get_tables(boxes)

            for table in tables:

                if len(table) > 1:
                    print(len(table))
                    boundary = table_cell_extraction.get_table_boundary(table)
                    cropped = img[boundary[1]: boundary[1] + boundary[3], boundary[0]: boundary[0] + boundary[2]]
                    # cv2.imshow(str(boundary), cropped)
                    # cv2.waitKey(0)
                    print(table)

                    # if len(table) == 77:
                    #     for cell in table:
                    #         cv2.imshow('name', img[cell[1]: cell[1] + cell[3], cell[0]: cell[0] + cell[2]])
                    #         cv2.waitKey(0)
                    if paths is None:
                        vectors, association, kv, parsed_strings = vectorization.table_to_vectors(table, image_path, document, embeddings_dict,
                                                                          average_vec, pre_jpg=True)
                    else:
                        vectors, association, kv, parsed_strings = vectorization.table_to_vectors(table, image_path, document, embeddings_dict,
                                                                          average_vec, pre_jpg=False)
                    data_point = [vectors, association, kv, parsed_strings]
                    parsed_data.append(data_point)

    print(parsed_data)

    ranking_model = TableKVExtractionCode.ranking_model.ranking_model(54, 120)
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(ranking_model.parameters(), lr=0.001)

    epoch_timestamp = datetime.datetime.now()

    for epoch in range(num_epochs):
        epoch_loss = 0
        ranking_model.train()

        for data in parsed_data:
            vectors = data[0]
            association = data[1]
            vectors_tensor = torch.from_numpy(np.hstack(vectors))
            # print(vectors)
            # print(association)
            # print(np.shape(vectors))
            # print(np.shape(association))

            for index, vector in enumerate(vectors):
                if association[index].unsqueeze(0).long() >= 0:
                    vector_tensor = torch.from_numpy(vector)
                    output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                    loss = loss_function(torch.unsqueeze(output, 0), association[index].unsqueeze(0).long())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    epoch_loss = epoch_loss + loss
        previous_time = epoch_timestamp
        epoch_timestamp = datetime.datetime.now()

        print("Epoch: " + str(epoch))
        print("Loss: " + str(epoch_loss))
        print("Epoch time: " + str(epoch_timestamp - previous_time) + ' ms')

    evaluate(ranking_model, parsed_data)


def evaluate(ranking_model, parsed_data):
    ranking_model.eval()
    correct_identified = 0
    incorrect_identified = 0
    hit2 = 0
    hit3 = 0
    num_documents = 0
    for data in parsed_data:
        num_documents += 1
        vectors = data[0]
        association = data[1]
        parsed_strings = data[3]
        vectors_tensor = torch.from_numpy(np.hstack(vectors))
        for index, vector in enumerate(vectors):
            if association[index].unsqueeze(0).long() >= 0 and parsed_strings[association[index].unsqueeze(0).long()] is not None:
                vector_tensor = torch.from_numpy(vector)
                output = ranking_model(vector_tensor.float(), vectors_tensor.float())
                output[output == 1] = 0
                output[parsed_strings is None] = 0
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

    #
    # loss = loss_function(torch.unsqueeze(output, 0), association[index].unsqueeze(0).long())
    # opt.zero_grad()
    # loss.backward()
    # opt.step()
    #
    # epoch_loss = epoch_loss + loss


# paths = [r"C:\Users\vchu1\Downloads\OCR_Sample_V1_SiliconExpert_confidential.xlsx - Connector Headers and PCB Recep.csv"]
# train(paths)
train()
