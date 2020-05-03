import os
import torch
import numpy as np
from torch.utils.data import Dataset


class trainData(Dataset):
    def __init__(self):
        super(trainData, self).__init__()
        # ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.

        voxData = np.load('data/modelnet10.npz')

        xTest = voxData['X_test']  # ndarray of size (908, 30, 30, 30)
        xTrain = voxData['X_train']  # ndarray of size (3991, 30, 30, 30)
        yTest = voxData['y_test']  # ndarray of size (908, )
        yTrain = voxData['y_train']  # ndarray of size (3991, )

        bcelabels = np.zeros((3991, 10))

        for i in range(len(yTrain)):
            idx = yTrain[i]
            bcelabels[i][idx] = 1

        self.xData = xTrain
        self.yData = yTrain
        self.BCELabels = bcelabels #Each label consists of a 10-length vector where the corresponding index of the desired object class is 1 and everything else is zero.

    def __getitem__(self, idx):
        return self.xData[idx], self.yData[idx], self.BCELabels[idx]

    def __len__(self):
        return len(self.yData)


if __name__ == "__main__":
    dataset = trainData()


class testData(Dataset):
    def __init__(self):
        super(testData, self).__init__()
        # ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.

        voxData = np.load('data/modelnet10.npz')
        # print(dataset.files)

        xTest = voxData['X_test']  # ndarray of size (908, 30, 30, 30)
        xTrain = voxData['X_train']  # ndarray of size (3991, 30, 30, 30)
        yTest = voxData['y_test']  # ndarray of size (908, )
        yTrain = voxData['y_train']  # ndarray of size (3991, )

        self.xData = xTest
        self.yData = yTest

    def __getitem__(self, idx):
        return self.xData[idx], self.yData[idx], self.testLabels[idx]

    def __len__(self):
        return len(self.yData)


if __name__ == "__main__":
    dataset = trainData()
