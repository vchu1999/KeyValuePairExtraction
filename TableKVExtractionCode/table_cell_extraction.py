import cv2
import numpy as np
import pytesseract
from collections import Counter


def vertical_overlap(b1, b2):
    """
    Checks if there is horizontal overlap between two bounding boxes
    :param b1: (x, y, width, height) of the first box
    :param b2: (x, y, width, height) of the second box
    :return: boolean if the two boxes overlap horizontally
    """
    y1, height1 = b1[1], b1[3]
    y2, height2 = b2[1], b2[3]

    if y2 <= y1 <= y2 + height2 or y2 <= y1 + height1 <= y2 + height2:
        return True

    return False


def horizontal_overlap(b1, b2):
    """
    Checks if there is vertical overlap between two bounding boxes
    :param b1: (x, y, width, height) of the first box
    :param b2: (x, y, width, height) of the second box
    :return: boolean if the two boxes overlap vertically
    """
    x1, width1 = b1[0], b1[2]
    x2, width2 = b2[0], b2[2]

    if x2 <= x1 <= x2 + width2 or x2 <= x1 + width1 <= x2 + width2:
        return True

    return False


def overlap(b1, b2):
    """
    Checks if two bounding boxes overlap
    :param b1: (x, y, width, height) of the first box
    :param b2: (x, y, width, height) of the second box
    :return: boolean if the two boxes overlap
    """
    return horizontal_overlap(b1, b2) and vertical_overlap(b1, b2)


def find_box_rows(boxes):
    """
    Gets rows of bounding boxes
    Not used for arrow data but can be useful
    :param boxes: list of bounding boxes in (x, y, width, height) format
    :return: list of lists of rows of bounding boxes
    """
    # TODO: fix with BFS
    out = []
    for box in boxes:
        found = False
        for level in out:
            if horizontal_overlap(box, level[0]):
                level.append(box)
                found = True

        if not found:
            out.append([box])

    return out


def get_box_images(img, boxes):
    """
    Gets cropped images from original image given bounding boxes
    :param img: original image
    :param boxes: list of bounding boxes
    :return: list of images of the bounding boxes
    """
    images = []
    for box in boxes:
        x, y, w, h = box
        cropped = img[y:y + h, x:x + w]
        images.append(cropped)
    return images


def get_horizontal_lines(img):
    """
    Finds most prominent horizontal lines
    @param img: input image
    @return: image with the most prominent horizontal lines
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # The kernel size affects the minimum thickness a line should have to be detected.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    return detect_horizontal


def get_vertical_lines(img):
    """
    Finds most prominent vertical lines
    @param img: input image
    @return: image with the most prominent vertical lines
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # The kernel size affects the minimum thickness a line should have to be detected.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    return detect_vertical


def sort_contours(cnts, method="left-to-right"):
    """
    Function to sort contours using given method
    taken from jestaban and portia github
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes),
                                       key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes


def get_cells(img):
    """
    Finds rectangular regions of interest (cells)
    Cells are table images of horizontal and vertical lines
    Cells are defined by their (x-coord of top left corner, y-coord of top left corner, width, height)
    @param img: image of horizontal and vertical lines
    @return: (image of boxes, cell information)
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, bounding_boxes = sort_contours(contours, method="top-to-bottom")

    boxes = []

    output_image = np.zeros(img.shape)
    output_image.fill(255)
    for c in contours:
        # Get position (x,y), width and height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # show contours on image
        if w < 450 and h < 35:
            output_image = cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append((x, y, w, h))
    return output_image, boxes


def get_tables(boxes, threshold=5):
    """
    Determines tables from boxes detected in an image
    Groups neighboring cells into tables
    :param boxes: (cells) defined by their (x-coord of top left corner, y-coord of top left corner, width, height)
    :param threshold: maximum distance between two cells to still be considered neighbors, default 5
    :return: (list of tables, dictionary of neighbors of cells)
    """
    cell_neighbors = {}

    for index, box in enumerate(boxes):
        # initialize box in question as a key
        if box not in cell_neighbors.keys():
            cell_neighbors[box] = set()

        # for the rest of the boxes, see if they neighbor the box in question
        for j in range(index + 1, len(boxes)):
            box_to_compare = boxes[j]
            if neighbor(box, box_to_compare, threshold):
                cell_neighbors[box].add(box_to_compare)

                # add the box in question as a neighbor to the box being compared
                if box_to_compare in cell_neighbors.keys():
                    cell_neighbors[box_to_compare].add(box)
                else:
                    cell_neighbors[box_to_compare] = {box}

    # List of tables. A table is represented by a set of its cell coordinates.
    tables = []

    # BFS to find groupings of cells, which are tables
    visited = set()
    queue = []
    for box in boxes:
        if box not in visited:
            visited.add(box)
            new_table = {box}
            for neighbor_box in cell_neighbors[box]:
                queue.append(neighbor_box)
            while len(queue) > 0:
                next_box = queue.pop(0)
                if next_box not in visited:
                    visited.add(next_box)
                    new_table.add(next_box)
                    for neighbor_boxes in cell_neighbors[next_box]:
                        if neighbor_boxes not in visited:
                            queue.append(neighbor_boxes)
            tables.append(new_table)
    return tables, cell_neighbors


def neighbor(box1, box2, threshold):
    """
    Checks if two cells are neighboring
    For checking if they should be grouped together into a table.
    :param box1: defined by (x-coord of top left corner, y-coord of top left corner, width, height)
    :param box2: defined by (x-coord of top left corner, y-coord of top left corner, width, height)
    :param threshold: maximum distance between two cells to still be considered neighbors
    :return: boolean if they are neighbors
    """
    # box1 too far right; box1 too far left; box1 too far up; box2 too far down
    if ((box1[0] - threshold > box2[0] + box2[2]) or
            (box1[0] + box1[2] + threshold < box2[0]) or
            (box1[1] - threshold > box2[1] + box2[3]) or
            (box1[1] + box1[3] + threshold < box2[1])):
        return False
    else:
        return True


def threshold_iterator(path, low, high):
    """
    Iterates through different possible values for input threshold for determining whether two cells are neighbors
    For manual evaluation for finding the best value for the threshold
    Iterates through the integers
    Prints the integer threshold and number of tables detected
    :param path: path to image
    :param low: integer lower bound for the threshold
    :param high: integer upper bound for the threshold
    :return: void
    """
    img = cv2.imread(path)

    horizontal = get_horizontal_lines(img)
    vertical = get_vertical_lines(img)
    lines = horizontal + vertical

    cell_image, boxes = get_cells(lines)

    for threshold in range(low, high + 1):
        tables = get_tables(boxes, threshold)
        print("Threshold: " + str(threshold))
        print("Number of tables: " + str(len(tables)))


def get_key_value_pairs(images, path_to_tesseract):
    """
    Gets key-value pairs from cropped cell images
    Only extracts key and value when they exist in the same cell
    Assumes key-value pairs arranged horizontally
    Prints key-value pairs
    :param images: list of images
    :param path_to_tesseract: raw path to tesseract
    :return: void
    """
    splits = []

    for index, image in enumerate(images):
        pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
        text = pytesseract.image_to_string(image)

        # remove junk from non descriptive cells
        split = [line for line in text.split('\n') if
                 line.strip() != '' and '®' not in line and any(c.isalnum() for c in line)]
        for s in split:
            if s.isspace():
                split.remove(s)

        splits.append(split)

    # get keys
    keys = [k[0] for k in splits if len(k) > 1]
    keys_counter = Counter(keys)

    # print values for keys
    for split in splits:
        if len(split) > 1:
            key = split[0]
            value = ' '.join(split[1:])
            if not key[0].isdigit() and keys_counter[key] < 2:
                print("key:", key, "\nval:", value)


def extract(path):
    """
    Prints key-value pairs algorithmically
    @param path: path to image
    @return: void, prints key-value pairs for the image
    """
    img = cv2.imread(path)

    horizontal_lines = get_horizontal_lines(img)
    vertical_lines = get_vertical_lines(img)
    lines = horizontal_lines + vertical_lines

    cell_image, boxes = get_cells(lines)

    images = get_box_images(img, boxes)

    print("Key/Values for " + path.strip('.jpg'))
    get_key_value_pairs(images)


def get_table_boundary(table):
    """
    Finds outer boundary for a table
    For showing an extracted table for a demo or as a part of table extraction pipeline.
    :param table: list of cells (x-coord of top left corner, y-coord of top left corner, width, height)
    :return: box that encompasses entire table (x-coord of top left corner, y-coord of top left corner, width, height)
    """
    first = True

    for cell in table:
        if first:
            x_low = cell[0]
            x_high = cell[0] + cell[2]
            y_low = cell[1]
            y_high = cell[1] + cell[3]
            first = False

        else:
            new_x_low = cell[0]
            new_x_high = cell[0] + cell[2]
            new_y_low = cell[1]
            new_y_high = cell[1] + cell[3]
            if new_x_low < x_low:
                x_low = new_x_low
            if new_x_high > x_high:
                x_high = new_x_high
            if new_y_low < y_low:
                y_low = new_y_low
            if new_y_high > y_high:
                y_high = new_y_high

    boundary = (x_low, y_low, x_high - x_low, y_high - y_low)
    return boundary


def sort_boxes(boxes):
    """
    Sorts bounding boxes by precedence (left to right and top to bottom)
    Not used for arrow data but can be useful
    :param boxes: list of bounding boxes in (x, y, width, height) format
    :return: list of sorted bounding boxes
    """
    # first finds rows of boxes to sort vertically and then sorts each row left to right
    rows = find_box_rows(boxes)
    out = []
    for row in rows:
        row.sort(key=lambda x: x[0])
        out.extend(row)
    return out
