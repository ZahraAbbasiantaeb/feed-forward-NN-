import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

def load_data():

    mnist = input_data.read_data_sets("/tmp/data/")

    train_x = mnist.train.images[0:44]

    validation_x = mnist.train.images[44:55]

    label_validation = mnist.train.labels[44:55]

    label_train = mnist.train.labels[0:44]

    test_x = mnist.test.images

    label_test = mnist.test.labels

    train_x = convertToDF(train_x)

    validation_x = convertToDF(validation_x)

    label_validation = convertToDF_oneD(label_validation)

    label_validation.col1 = label_validation.col1.astype(int)

    label_train = convertToDF_oneD(label_train)

    label_train.col1 = label_train.col1.astype(int)

    test_x = convertToDF(test_x)

    label_test = convertToDF_oneD(label_test)

    label_test.col1 = label_test.col1.astype(int)

    label_test = label_test.pop('col1')

    label_train = label_train.pop('col1')

    return train_x, label_train, test_x, label_test, validation_x, label_validation


def convertToDF(dataset):

    index = ['Row' + str(i) for i in range(1, len(dataset) + 1)]

    key = ['col' + str(i) for i in range(1, len(dataset[0]) + 1)]

    dataFrame = pd.DataFrame(dataset, index=index, columns =key)

    dataFrame.keys()

    return dataFrame


def convertToDF_oneD(dataset):

    index = ['Row' + str(i) for i in range(1, len(dataset) + 1)]

    key = ['col1']

    dataFrame = pd.DataFrame(dataset, index=index, columns=key)

    dataFrame.keys()

    return dataFrame

