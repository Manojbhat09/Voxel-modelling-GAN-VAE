import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # Layer 1 - conv3d - Input: 32x32x32, Output:30x30x30
        # Layer 2 - conv3d - Input: 30x30x30, Output:15x15x15
        # Layer 3 - conv3d - Input: 15x15x15, Output:13x13x13
        # Layer 4 - conv3d - Input: 13x13x13, Output:7x7x7
        # Layer 5 - Linear - Input: 7x7x7, Output:343
        # encMu - Linear - Input: 343, Output: 100 (Latent Dimension; needs to be reparameterized)
        # encStd - Linear - Input: 343, Output: 100 (Latent Dimension; needs to be reparameterized)

        latent_dim = 100
        self.encMu = nn.Linear(343, latent_dim)
        self.encStd = nn.Linear(343, latent_dim)

        # Main throughput is be a 5d tensor - (batchSize, numChannels, Depth, Height, Width)
        self.encModel = nn.Sequential(
            nn.Conv3d(1, 2, (3,3,3), stride=(1,1,1),padding=0), #inchannel, outchannel, kernelSize, stride, padding
            nn.ELU(),
            nn.BatchNorm3d(2), #Expects a 5d input (batchSize, numChannels, Depth, Height, Width). Arg is numChannels.
            nn.Conv3d(2, 3, (3,3,3), stride=(2,2,2),padding=1), #Layer 2
            nn.ELU(),
            nn.BatchNorm3d(3),
            nn.Conv3d(3, 4, (3,3,3), stride=(1,1,1),padding=0), #Layer 3
            nn.ELU(),
            nn.BatchNorm3d(4),
            nn.Conv3d(4, 5, (3,3,3), stride=(2,2,2),padding=1), #Layer 4
            nn.ELU(),
            nn.BatchNorm3d(5),
            #nn.Linear(64*7*7*7, 343), #Layer 5 - Input: (batchSize, channels=64, 7,7,7), Output: (batchSize, channels=64, 343)
            #nn.ELU(),
            #nn.BatchNorm1d(64) #Expects a 3d input (batchSize, numChannels=64, 343)
        )

        self.fc1 = nn.Linear(5*6*6*6, 343)
    
    def forward(self, x):
        #x should be of dim (batchSize, numChannels, Depth, Height, Width)
        out1 = self.encModel(x)

        out2 = out1.reshape(x.size(0), -1)
        latent_input = self.fc1(out2)

        predMu = self.encMu(latent_input)
        predStd = self.encStd(latent_input)

        return predMu, predStd


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #Are there 64 latents, or just 1 that needs to be converted to 64 channels?

        #Layer 1 - linear - Input: 343, Output:7x7x7
        #Layer 2 - conv3d - Input: 7x7x7, Output: 15x15x15
        #Layer 3 - conv3d - Input: 15x15x15, Output: 15x15x15
        #Layer 4 - conv3d - Input: 15x15x15, Output: 32x32x32
        #Layer 5 - conv3d - Input: 32x32x32, Output: 32x32x32 (Output)

        self.batchDim = 307

        self.decModel = nn.Sequential(
            #nn.Linear(343, (7,7,7)),
            #nn.ELU(),
            #nn.BatchNorm3d(64), #Expects a 5d input (batchSize, numChannels=64, 7,7,7)
            nn.Conv3d(5, 4, (3,3,3), stride=(1,1,1),padding=0), #Layer 2
            nn.ELU(),
            nn.BatchNorm3d(4),
            nn.Conv3d(4, 3, (3,3,3), stride=(1,1,1),padding=1), #Layer 3 #Check Padding
            nn.ELU(),
            nn.BatchNorm3d(3),
            nn.Conv3d(3, 2, (4,4,4), stride=(1,1,1),padding=0), #Layer 4
            nn.ELU(),
            nn.BatchNorm3d(2),
            nn.Conv3d(2, 1, (3,3,3), stride=(1,1,1),padding=1), #Layer 5 #Check Padding
            #nn.BatchNorm3d(1)
        )


    def forward(self, x): #x is the latent layer, could be random noise. Is a probabilistic distribution.

        self.fc1 = nn.Linear(100, 5*6*6*6)

        #x is of dim batch_dim x 100

        #print(type(x))
        #print(x.shape)

        #out2 = out1.reshape(x.size(0), -1)

        out1 = self.fc1(x)
        out2 = out1.view(self.batchDim, 5, 6, 6, 6)

        #print(type(out2))
        #print(out2.shape)

        out3 = self.decModel(out2)

        return out3

class VAE(nn.Module):
    def __init__(self, object_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(latent_dim)
        self.dec = Decoder()

    def forward(self, x):
        mu, std = self.enc.forward(x) # x has dimensions batchSize, object_dim 
        #reparameterize
        std = torch.exp(0.5*std)
        eps = torch.randn_like(std)
        z = mu + eps*std 
        output = self.dec.forward(z) 
        return output, mu, std
    def decode(self, z):
        # given random noise z, generates a voxel object
        return self.dec(z)

