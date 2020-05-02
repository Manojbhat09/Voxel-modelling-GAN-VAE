import os
import torch
import numpy as np
from torch.utils.data import Dataset


class VoxelDataSet(Dataset):

    def __init__(self, path='./voxel_data'):
        super(VoxelDataSet, self).__init__()
        self.voxels = np.array() # Storage for voxel objects
        voxels_fn = [afn for afn in os.listdir(path) if afn.endswith('.dat')]

        # Get Voxel Arrays
        initY = False
        initX = False
        initV = False
        with open(os.path.join(path, voxels_fn[0]), 'r', encoding="utf8", errors='ignore') as f:
            raw_data = f.readlines()

            voxel_ph = 0
            for obj in range(len(raw_data)):
                current_voxel = raw_data[obj].split(' ')

                while "" in current_voxel:
                    current_voxel.remove("")
                #This schema assumes the data lists, in order, each z value for a given y, 
                # for each y at a given x, for each x
                for x in range(32):
                    for y in range (32):
                        temp_1d = []
                        for z in range(32):
                            temp_1d.append(current_voxel.pop(0)) #List of z values for a given y and x
                        
                        np_1d = np.array(temp_1d) #Z values for a given y, 1-D

                        if initY == False:
                            np_2d = np.array(temp_1d)
                            initY = True
                        else:
                            np.concatenate(np_2d, np_1d) #Assembles z values for each y at a given x, 2-D
                    
                    initY = False #Resets z row storage on next run

                    if initX == False:
                        np_3d = np_2d
                        initX = True
                    else:
                        np.concatenate(np_3d, np_2d) #Assembles zy planes for each x, 3-D
                
                initX = False #Resets zy plane storage on next run

                if initV == False:
                    voxel_ph = np_3d
                    initV = True
                else:
                    np.concatenate(voxel_ph, np_3d)

        self.voxels = voxel_ph #4 dimensional np array

    def __getitem__(self, idx):
        return self.voxels[idx]
        
    def __len__(self):
        return len(self.voxels)

class trainData(Dataset):
    def __init__(self):
        super(trainData, self).__init__()
        # ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.

        voxData = np.load('modelnet10.npz')

        xTest = voxData['X_test']  # ndarray of size (908, 30, 30, 30)
        xTrain = voxData['X_train']  # ndarray of size (3991, 30, 30, 30)
        yTest = voxData['y_test']  # ndarray of size (908, )
        yTrain = voxData['y_train']  # ndarray of size (3991, )

        self.xData = xTrain
        self.yData = yTrain

    def __getitem__(self, idx):
        return self.xData[idx], self.yData[idx]

    def __len__(self):
        return len(self.yData)


if __name__ == "__main__":
    dataset = trainData()


class testData(Dataset):
    def __init__(self):
        super(testData, self).__init__()
        # ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.

        voxData = np.load('modelnet10.npz')
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
