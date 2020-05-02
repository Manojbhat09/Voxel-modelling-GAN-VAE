import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from SegFault_DataSet import VoxelDataSet
from SegFault_VAE import VAE
from utils import *

from torch.nn import functional as F


def main():
    device = 'cpu'

    # define dataset and dataloader
    dataset = VoxelDataSet()
    voxel_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Definer Hyperparameters
    latent_dim = 100
    lr = 0.005         # learning rate
    num_epochs = 100

    # Build the model
    vae = VAE(object_dim=32, latent_dim=latent_dim).to(device)
    print("VAE model:\n", vae)

    #Loss Function
    def getLoss(reconstructedData, data, mu, std):

        MSELoss = F.mse_loss(reconstructedData, data)

        KLDivergence = -0.5 * torch.sum(1 + std - mu.pow(2) - std.exp())

        a = 3 #alpha, weight of KL Divergence

        loss = MSELoss + a * KLDivergence

        return loss

    # define optimizer for discriminator and generator separately
    optim = Adam(vae.parameters(), lr=lr)

    epochVals = []
    lossVals = []
    
    #Training
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(voxel_dataloader):
            voxel_labels = local_batch.to(device)

            output, mu, std = vae(voxel_labels) 

            loss = getLoss(output, voxel_labels, mu, std)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

        epochVals = epochVals + [epoch]
        lossVals = lossVals + [loss]

    torch.save(vae.state_dict(), 'vae_model.ckpt')

    plt.figure()
    for i in range(len(epochVals)):
        plt.plot(epochVals, lossVals)
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.show()

    #Reconstruct Objects, Output Text File


if __name__ == "__main__":
    main()