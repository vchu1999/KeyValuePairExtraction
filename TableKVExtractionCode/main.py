#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andrewkreisher
"""
from TableKVExtractionCode import utils


# for command line
# prints out results, can change to exporting to .txt file pretty easi
def main(path, demo, table):
    # utils.text_location_extract()
    # paths = utils.pdf_to_image(path)
    paths = [path]

    for path in paths:
        if table:
            # print(utils.neighbor([2667, 1509, 148, 53], [2817, 1454, 187, 53], 5))
            utils.demo_table(path, True, True)
            # utils.threshold_iterator(path, 0, 100)
        else:
            if demo:
                utils.demo(path, True, True)
            else:
                utils.extract(path)


if __name__ == '__main__':
    # path = sys.argv[1]
    # demo = False
    # if len(sys.argv) > 2:
    #     if sys.argv[2] == '-d':
    #         demo = True
    # main(path, demo)
    # main(r"C:\Users\vchu1\Downloads\iec-nh-4-np-ob-rotated.pdf", True)
    # main(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\887downloader.phppsd-pmaxxx-mmtbl-cxx-xft.pdf", True)
    # main(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\sl9126n-osra.pdf", True)

    # main(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\arrowtest.pdf", True, True)
    # main(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\1200-8125-
    # main(r"C:\DocumentProcessingLab\arrow-electronics-data-extraction\2wb1152-xx_a9.pdf", True, True)
    # main(r"C:\DocumentProcessingLab\arrow_electronics_data_extraction\arrowtest.pdf",True, True)
    # main(r"C:\DocumentProcessingLab\arrow_electronics_data_extraction\arrowtest.pdf",True, True)
    main(r"D:\AGuptaKVPairExtraction\TableKVExtractionCode\generated_documents\         1.jpg",True, True)










