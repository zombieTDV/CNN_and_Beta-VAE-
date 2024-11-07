"""
Variational Autoencoder (VAE) implementation for MNIST dataset

This script implements a Variational Autoencoder (VAE) to reconstruct MNIST digits.
It uses a custom deep learning framework (SUPER_AVK) for tensor operations and neural network layers.

The VAE consists of an encoder, a latent space, and a decoder. It is trained using the ELBO loss,
which combines reconstruction error (RMSE) and KL divergence.
"""

from SUPER_AVK.AvK import Tensor, concat
from SUPER_AVK.Layers import Convolution2D, Linear, Transpose_Convolution2D_RT, Layer_Norm
from SUPER_AVK.Optimizer_Module import Optimizer
from SUPER_AVK.Utility_functions import lr_scheduler
 
import idx2numpy
import numpy as np
import matplotlib_animator
import matplotlib.pyplot as plt

# Uncomment for profiling
# from cProfile import Profile
# from pstats import SortKey, Stats
# np.set_printoptions(4, suppress=True)

# Define paths for MNIST dataset files
dtrain = 'data/train-images.idx3-ubyte'
dtrain_label = 'data/train-labels.idx1-ubyte'
dvalid = 'data/t10k-images.idx3-ubyte'
dvalid_label = 'data/t10k-labels.idx1-ubyte'

# Load and preprocess MNIST dataset
train = ((idx2numpy.convert_from_file(dtrain)/255).reshape(-1, 1, 28, 28))
train_label = idx2numpy.convert_from_file(dtrain_label)
valid = ((idx2numpy.convert_from_file(dvalid)/255).reshape(-1, 1, 28, 28))
valid_label = idx2numpy.convert_from_file(dvalid_label)

class VAE:
    """
    Variational Autoencoder (VAE) implementation
    """
    def __init__(self, inp_shape, laten_space_dims = 24):
        """
        Initialize the VAE architecture
        
        Args:
            inp_shape (tuple): Input shape (C, H, W)
            laten_space_dims (int): Dimensionality of the latent space
        """
        # Initialize VAE architecture
        self.C, self.H, self.W = inp_shape
        self.laten_space_dims = laten_space_dims
        
        # Encoder layers
        self.CNN1 = Convolution2D(inp_shape, 32, 3, 1, 0, pooling=True, nonliner=True, batch_norm=True, act_func='leaky relu')
        self.CNN2 = Convolution2D(self.CNN1.out_shape, 64, 2, 1, 0, pooling=True, nonliner=True, batch_norm=True, act_func='leaky relu')
        self.CNN3 = Convolution2D(self.CNN2.out_shape, 64, 3, 1, 0, pooling=False, nonliner=True, batch_norm=True, act_func='leaky relu')
        
        # Latent space layers
        self.MEAN_layer = Linear(np.prod(self.CNN3.out_shape), self.laten_space_dims, req_bias=True,
                                 initialization='standard normal', act_func='leaky relu')
        self.LOG_VAR_layer = Linear(np.prod(self.CNN3.out_shape), self.laten_space_dims, req_bias=True,
                                    initialization='standard normal', act_func='leaky relu')
        
        # Decoder layers
        self.Dense = Linear(self.laten_space_dims, np.prod(self.CNN3.out_shape))
        self.TCNN3 = Transpose_Convolution2D_RT(self.CNN3.out_shape, self.CNN3, nonliner=True, upsampling = False, batch_norm=True,
                                                act_func='leaky relu')
        self.TCNN2 = Transpose_Convolution2D_RT(self.TCNN3.out_shape, self.CNN2, nonliner=True, upsampling = True, batch_norm=True,
                                                act_func='leaky relu')
        self.TCNN1 = Transpose_Convolution2D_RT(self.TCNN2.out_shape, self.CNN1, nonliner=True, upsampling = True, batch_norm=True,
                                                act_func='sigmoid')
        
        # Normalization layer
        self.LayerNorm = Layer_Norm(C = self.Dense.Weights.shape[1])

    def forward(self, X, mode):
        """
        Forward pass through the VAE
        
        Args:
            X (Tensor): Input tensor
            mode (str): 'train' or 'inference'
        
        Returns:
            Tensor: Reconstructed output
        """
        B = X.shape[0]
        # Encoder
        conv1 = self.CNN1(X, mode)
        conv2 = self.CNN2(conv1, mode)
        conv3 = self.CNN3(conv2, mode)
        
        # Latent space
        self.MU = self.MEAN_layer(conv3.reshape(B, -1))
        self.LOG_VAR = self.LOG_VAR_layer(conv3.reshape(B, -1))
        SIGMA = (self.LOG_VAR / 2).exp()
        
        # Reparameterization trick
        standard_norm_dist = np.random.randn(B, self.laten_space_dims)
        sampled_points_in_latent_space = self.MU + SIGMA * standard_norm_dist
        
        # Decoder
        linear = self.Dense(sampled_points_in_latent_space)
        norm = self.LayerNorm(linear)
        reshape = norm.reshape(B, 64, 4, 4)
        tconv3 = self.TCNN3(reshape, mode)
        tconv2 = self.TCNN2(tconv3, mode)
        tconv1 = self.TCNN1(tconv2, mode)
        
        reconstruction_output = tconv1.reshape(B, 1, 28, 28)
        return reconstruction_output

# Training loop
optimizer = Optimizer(lr=0.1, optimization_method = 'AdamW')
B = 32  # Batch size
VAE = VAE((1,28,28))

RMSE_loss = []
D_KL_loss = []
LOSS = []
epochs = 1000
for i in range(epochs):
    # Sample random batch
    rand_batch = np.random.randint(low=0, high=len(train.data), size=B)
    X = Tensor(train[rand_batch, ], is_parameters=False)
    
    # Forward pass
    hidden = VAE.forward(X, mode='train')
    
    # Calculate loss
    Beta = 1e-6  # KL divergence weight
    RMSE, D_KL, loss = hidden.ELBO(VAE.MU, VAE.LOG_VAR, X, Beta)
    
    # Backward pass and optimization
    loss.backward(optimizer=lr_scheduler(optimizer, epochs, 0.05, decay_step=500), update=True, retain_all_grad=False)
    
    # Print and store loss values
    if i % 50 == 0:
        print(f'{i=}\t{RMSE.data=}\t{D_KL.data*Beta=}\t{D_KL.data=}\t{loss.data=}')
        RMSE_loss.append(RMSE.data)
        D_KL_loss.append(D_KL.data*Beta)
        LOSS.append(loss.data)
        
    loss.clean_neural_net()
    # print(hidden.grad)
# Uncomment to plot loss curves
n = 0
plt.plot(np.arange(len(RMSE_loss)-n), RMSE_loss[n:], label='RMSE', linestyle="-")
plt.plot(np.arange(len(RMSE_loss)-n), D_KL_loss[n:], label='D_KL', linestyle="-.")
plt.plot(np.arange(len(RMSE_loss)-n), LOSS[n:], label='LOSS', linestyle=":")
plt.legend()
plt.show()

# Visualization of results
animator = matplotlib_animator.ANIMATOR("Recon IMG")
b = 9
n = np.random.randint(0,10000, 9)

img = Tensor(valid[n])
recon_img = VAE.forward(img, 'inferance')

animator.multiple_img([x for x in concat(list(img) + list(recon_img)).data[:,0]], 9, 2, cmap='grey')
