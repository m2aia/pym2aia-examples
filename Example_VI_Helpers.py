import numpy as np
import torch
from torch import nn

# 2.2) define the loss function
#===================================
class VAE_Loss(nn.Module):
    def __init__(self, spectrum_depth):
        super().__init__()
        self.spectrum_depth = spectrum_depth
        self.MSE = nn.MSELoss()
        
    def forward(self, inputs, outputs, z_mean, z_log_var):
        kl_Loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
        kl_Loss = torch.sum(kl_Loss, axis=-1)
        kl_Loss *= -0.5
        
        # compare center spectrum with reconstructed spectrum (output)
        reconstruction_loss = self.MSE.forward(inputs[:,0,:,1,1],outputs) 
        reconstruction_loss *= self.spectrum_depth
        
        # ========== Compile VAE_BN model ===========
        model_Loss = torch.mean(reconstruction_loss + kl_Loss)
        return model_Loss

# 2.3) Variational Auto Encoder
#===================================
class VAE(nn.Module):           
    
    class Reparameterization(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()

            self.z_mean = nn.Linear(input_size, output_size)
            self.z_log_var =  nn.Linear(input_size, output_size)

        def forward(self, X):
            z_mean = self.z_mean(X)
            z_log_var = self.z_log_var(X)
            batch = z_mean.shape[0]
            dim = z_mean.shape[1]
            epsilon = torch.normal(torch.zeros((batch, dim)), torch.ones((batch, dim))).cuda() # random_normal (mean=0 and std=1)
            z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
            return z, z_mean, z_log_var

    class Reshape(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def forward(self, x):
            return x.view(self.shape)

    def __init__(self, spectrum_size, interim_dim, latent_size=64):
        super().__init__()

        d0 = spectrum_size
        d1 = np.floor((d0-1*(5-1)-1)/2) + 1
        d2 = np.floor((d1-1*(5-1)-1)/2) + 1
        d3 = np.floor((d2-1*(3-1)-1)/2) + 1
        d4 = int(np.floor((d3-1*(3-1)-1)/2) + 1)

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(5,3,3), stride=(2,1,1)),
            nn.Conv3d(4, 8, kernel_size=(5,1,1), stride=(2,1,1)),
            nn.Conv3d(8, 16, kernel_size=(3,1,1), stride=(2,1,1)),
            nn.Conv3d(16, 16, kernel_size=(3,1,1), stride=(2,1,1)),
            nn.Flatten(start_dim=1),
            nn.Linear(16*d4, interim_dim),
            nn.ReLU(),
            self.Reparameterization(interim_dim, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, interim_dim),
            nn.ReLU(),
            nn.Linear(interim_dim, spectrum_size)
        )

    def forward(self, X):
        
        (z, z_mean, z_log_var) = self.encoder(X)
        X = self.decoder(z) 

        return X, z, z_mean, z_log_var