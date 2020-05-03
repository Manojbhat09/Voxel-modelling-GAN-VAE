import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from SegFault_DataSet import VoxelDataSet
from SegFault_VAE import VAE

from torch.nn import functional as F

from SegFault_DataSet import trainData
from SegFault_DataSet import testData


def main():
    device = 'cpu'

    # Definer Hyperparameters
    latent_dim = 1000
    lr = 0.005         # learning rate
    num_epochs = 0
    batch_dim = 307 #13 or 307
    gamma = 0.97

    #Load Data
    #dataset = VoxelDataSet()
    #voxel_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    trainDataSet = trainData()
    trainLoader = DataLoader(dataset=trainDataSet, batch_size=batch_dim, shuffle=True, num_workers=0)
    testDataSet = testData()
    testLoader = DataLoader(dataset=testDataSet, batch_size=batch_dim, shuffle=True, num_workers=0)

    # Build the model
    vae = VAE(object_dim=32, latent_dim=latent_dim).to(device)

    #Loss Function
    def getLoss(reconstructedData, data, mu, std):


        MSELoss = F.mse_loss(reconstructedData, data)


        #print("  MSE: ", MSELoss.item())

        #print("         std: ", std)
        #print("         mu: ", mu)

        #KLDivergence = -0.5 * torch.sum(1 + std - mu.pow(2) - std.exp())

        #print("  KLD: ", KLDivergence.item())

        #a = 1 #alpha, weight of KL Divergence

        #loss = MSELoss + a * KLDivergence
        loss = MSELoss

        #print("  Loss: ", loss.item())

        return loss

    def newLoss(generatedData, targetData):

        loss = 0

        return loss


    # define optimizer for discriminator and generator separately
    optim = Adam(vae.parameters(), lr=lr, weight_decay=1e-4)

    epochVals = []
    lossVals = []

    print("BEGIN TRAINING")

    #Training
    for epoch in range(num_epochs):
        for n_batch, (x_batch, labels, BCELabels) in enumerate(trainLoader): #x_batch are the actual voxel objects, the labels are the modelnet object classes, ints from 0-9. The labels aren't terribly important.
                                                                    
            #print("    Batch ", n_batch)

            voxel_labels = labels.to(device)
            x_batch = x_batch.to(device)

            #x_batch is of dimensions (batch_size, 30, 30, 30). Need to add a dimension for channels.

            x_batch = x_batch.unsqueeze(1) 
            x_batch = x_batch.float()

            output, mu, std, l_z = vae(x_batch) 

            print(output.shape)

            #Get Labels
            BCELabels = BCELabels.to(device)

            loss = getLoss(output, x_batch, mu, std)

            optim.zero_grad()
            loss.backward()
            optim.step()

            #print("     Loss, ", loss )
            
            if (n_batch + 1) % 4 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

        epochVals = epochVals + [epoch]
        lossVals = lossVals + [loss]
        print("Epoch ", epoch)

    print("TRAINING FINISHED")

    torch.save(vae.state_dict(), 'vae_model.ckpt')

    plt.figure()
    for i in range(len(epochVals)):
        plt.plot(epochVals, lossVals)
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.show()

    #Reconstruct Objects

    reconstructedVoxels = torch.empty(0, 1, 30, 30, 30)

    print("BEGIN RECONSTRUCTION")

    for n_batch, (x_batch, labels, BCELabels) in enumerate(trainLoader): 
        x_batch = x_batch.to(device)
        x_batch = x_batch.unsqueeze(1) 
        x_batch = x_batch.float()

        output, mu, std, l_z = vae(x_batch) 

        x_batch = x_batch.squeeze()

        reconstructedVoxels = torch.cat((reconstructedVoxels, output), 0)
        print("Batch ", n_batch)

    print(reconstructedVoxels.shape)

    #Write to File

    outF = open("Generated_Voxels.txt", "w")
    outF.write("Generated Voxel Data \n")

    for idx in range(3991)
        outF.write("[")


    outF.close()


if __name__ == "__main__":
    main()