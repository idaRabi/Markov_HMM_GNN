import xml.etree.ElementTree as ET
import numpy as np


filesNames = ["a", "e", "i", "o", "u"]


def get_converted_xml(file_name):
    files_location = "/Users/Momeda/Desktop/HW3/Dataset1/%s.xml"
    file_location = files_location % file_name
    tree = ET.parse(file_location)
    root = tree.getroot()
    xml_arr = []
    for trainingExample in root:
        if trainingExample.tag == 'gesture':
            continue
        training_arr = []
        for cord in trainingExample:
            training_arr.append(cord.attrib)

        xml_arr.append(training_arr)
    return xml_arr

def get_train_and_test(vowel_name):
    a_xml = get_converted_xml(vowel_name)

    test_set = []
    train_set = []
    minX = 10 ** 10
    minY = 10 ** 10
    maxX = 0
    maxY = 0
    for trainingExample in a_xml:
        recordTest = []
        recordTrain = []

        for index, cord in enumerate(trainingExample):
            cord['x'] = float(cord['x'])
            cord['y'] = float(cord['y'])

            if cord['x'] < minX:
                minX = cord['x']
            if cord['y'] < minY:
                minY = cord['y']

        for index, cord in enumerate(trainingExample):
            cord['x'] -= minX
            cord['y'] -= minY
            if cord['x'] > maxX:
                maxX = cord['x']
            if cord['y'] > maxY:
                maxY = cord['y']

        for index, cord in enumerate(trainingExample):
            cord['x'] = cord['x']/maxX
            cord['y'] = cord['y']/maxY
            if index % 2 == 0:
                recordTrain.append(np.array([cord['x'], cord['y']]))
            if index % 2 == 1:
                recordTest.append(np.array([cord['x'], cord['y']]))

        test_set.append(recordTest)
        train_set.append(recordTrain)
    return train_set, test_set

