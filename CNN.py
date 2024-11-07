"""
Convolutional Neural Network (CNN) Autoencoder for Image Reconstruction

This script implements a CNN-based autoencoder model for image reconstruction using the MNIST dataset.
It uses custom implementations of neural network layers and optimization from the SUPER_AVK package.

Key components:
- Data loading and preprocessing
- Model definition (MLP_model2 class)
- Training loop

Dependencies:
- SUPER_AVK: Custom neural network package
- idx2numpy: For reading IDX files
- numpy: For numerical operations
- matplotlib: For plotting (currently unused)
"""

from SUPER_AVK.AvK import Tensor
from SUPER_AVK.Layers import Convolution2D, Linear, Transpose_Convolution2D_RT, Layer_Norm
from SUPER_AVK.Optimizer_Module import Optimizer

import idx2numpy
import numpy as np
# import numba
import matplotlib.pyplot as plt
# import matplotlib_animator

from cProfile import Profile
from pstats import SortKey, Stats
np.set_printoptions(4, suppress=True)

dtrain = 'data/train-images.idx3-ubyte'
dtrain_label = 'data/train-labels.idx1-ubyte'

dvalid = 'data/t10k-images.idx3-ubyte'
dvalid_label = 'data/t10k-labels.idx1-ubyte'

train = Tensor((idx2numpy.convert_from_file(dtrain)/255).reshape(-1, 1, 28, 28))
train_label = idx2numpy.convert_from_file(dtrain_label)

valid = Tensor((idx2numpy.convert_from_file(dvalid)/255).reshape(-1, 1, 28, 28))
valid_label = idx2numpy.convert_from_file(dvalid_label)

class MLP_model2:
    """
    Autoencoder model for image reconstruction.
    
    This model uses convolutional layers for encoding, dense layers for latent representation,
    and transpose convolutional layers for decoding.
    """

    def __init__(self, inp_shape):
        """
        Initialize the model layers.
        
        Args:
            inp_shape (tuple): Input shape (channels, height, width)
        """
        self.C, self.H, self.W = inp_shape
        self.CNN1 = Convolution2D(inp_shape, 16, 3, 1, 0, pooling=True, nonliner=True, batch_norm=True)
        self.CNN2 = Convolution2D(self.CNN1.out_shape, 32, 2, 1, 0, pooling=True, nonliner=True, batch_norm=True)
        self.CNN3 = Convolution2D(self.CNN2.out_shape, 64, 3, 1, 0, pooling=False, nonliner=True, batch_norm=True)

        self.Dense1 = Linear(np.prod(self.CNN3.out_shape), 50, req_bias=True, nonliner=True, act_func='relu')
        self.Dense2 = Linear(50, 1024, req_bias=True, nonliner=True, act_func='relu')

        self.TCNN3 = Transpose_Convolution2D_RT(self.CNN3.out_shape, self.CNN3, nonliner=True, upsampling = False, batch_norm=True)
        self.TCNN2 = Transpose_Convolution2D_RT(self.TCNN3.out_shape, self.CNN2, nonliner=True, upsampling = True, batch_norm=True)
        self.TCNN1 = Transpose_Convolution2D_RT(self.TCNN2.out_shape, self.CNN1, nonliner=True, upsampling = True, batch_norm=True,
                                                act_func='sigmoid')
        

        self.LayerNorm1 = Layer_Norm(C = self.Dense1.Weights.shape[1])
        self.LayerNorm2 = Layer_Norm(C = self.Dense2.Weights.shape[1])

        
    def forward(self, X, mode):
        """
        Forward pass through the autoencoder.
        
        Args:
            X (Tensor): Input tensor
            mode (str): 'train' or 'eval' mode
        
        Returns:
            Tensor: Reconstructed output
        """
        B = X.shape[0]
        conv1 = self.CNN1(X, mode)
        conv2 = self.CNN2(conv1, mode)
        conv3 = self.CNN3(conv2, mode)

        linear1 = self.Dense1(conv3.reshape(B, -1)) #flatten
        norm1 = self.LayerNorm1(linear1)
        linear2 = self.Dense2(norm1)
        norm2 = self.LayerNorm2(linear2)

        reshape = norm2.reshape(B, 64, 4, 4)

        tconv3 = self.TCNN3(reshape, mode)
        tconv2 = self.TCNN2(tconv3, mode)
        tconv1 = self.TCNN1(tconv2, mode)
        
        reconstruction_output = tconv1.reshape(B, 1, 28, 28)
        return reconstruction_output
        

    # def decoder(self, X):
    #     x = self.UP1(X)
    #     x = self.cnn1_U(x) 
    #     x = self.UP2(x)
    #     return x
    
    # def forward(self, train):
    #     encoder_output = self.encoder(train)
    #     decoder_output = self.decoder(encoder_output.reshape(-1, 1, 8,8))
    #     return decoder_output

optimizer = Optimizer(lr=0.1, optimization_method = 'AdamW')
B = 32
Model = MLP_model2((1,28,28))

for i in range(200):
    # Random batch selection
    rand_batch = np.random.randint(low=0, high=len(train.data), size=B)
    X = train[rand_batch, ]
    
    # Forward pass
    hidden = Model.forward(X, mode='train')
    
    # Loss calculation and backpropagation
    loss = hidden.MSE(X)
    loss.backward(optimizer, update=True, retain_all_grad=False)
    
    # Print loss every 10 iterations
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.data}")