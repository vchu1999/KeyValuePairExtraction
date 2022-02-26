import cv2

"""
Shows a demo of identifying tables in a document.
"""


def demo_table(path, view_preprocess=True, view_tables=True):
    img = cv2.imread(path)
    horiz = get_horizontal_lines(img)
    vert = get_vertical_lines(img)
    lines = horiz + vert
    out, boxes = get_cells(lines)

    if view_preprocess:
        cv2.imshow("original", img)
        cv2.waitKey(0)
        cv2.imshow("horiz", horiz)
        cv2.waitKey(0)
        cv2.imshow("vert", vert)
        cv2.waitKey(0)
        cv2.imshow("lines", lines)
        cv2.imwrite("lines.png", lines)

        cv2.waitKey(0)
        cv2.imshow("cells", out)
        cv2.imwrite("cells.png", out)

        cv2.waitKey(0)
        images = get_box_images(img, boxes)
        # for i, image in enumerate(images):
        #     cv2.imshow(str(i), image)
        #     cv2.waitKey(0)

    tables = get_tables(boxes)
    for table in tables:
        boundary = get_table_boundary(table)
        print("Table: " + str(table))
        print("Its boundary: " + str(boundary))
        cropped = img[boundary[1]: boundary[1] + boundary[3], boundary[0]: boundary[0] + boundary[2]]
        if view_tables:
            cv2.imshow(str(boundary), cropped)
            cv2.waitKey(0)
    print("Total number of tables is: " + str(len(tables)))


"""
Demos key-value extraction pipeline (algorithmic, not machine learning based)
"""


def demo(path, view_preprocess=True, view_cells=True):
    img = cv2.imread(path)
    horiz = get_horizontal_lines(img)
    vert = get_vertical_lines(img)
    lines = horiz + vert
    out, boxes = get_cells(lines)

    if view_preprocess:
        cv2.imshow("original", img)
        cv2.waitKey(0)
        cv2.imshow("horiz", horiz)
        cv2.waitKey(0)
        cv2.imshow("vert", vert)
        cv2.waitKey(0)
        cv2.imshow("lines", lines)
        cv2.imwrite("lines.png", lines)

        cv2.waitKey(0)
        cv2.imshow("cells", out)
        cv2.imwrite("cells.png", out)

        cv2.waitKey(0)
    images = get_box_images(img, boxes)
    if view_cells:
        for i, image in enumerate(images):
            cv2.imshow(str(i), image)
            cv2.waitKey(0)
    get_key_value_pairs(images)
