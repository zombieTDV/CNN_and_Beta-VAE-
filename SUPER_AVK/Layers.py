"""
Neural Network Layer Implementations

This module contains custom implementations of various neural network layers and utilities.
These layers are designed to work with the SUPER_AVK package for building and training
neural networks.

Key components:
- Initialization functions
- Linear layers
- Normalization layers (Layer Norm, Batch Norm)
- Dropout
- Convolutional layers (standard and transposed)
- Pooling layers
- Upsampling
- Attention mechanisms (Self-Attention, Multi-Head Attention)
- Beam search utilities
- Positional encoding
- Advanced convolutional layers (Convolution2D, Transpose_Convolution2D, Transpose_Convolution2D_RT)

Classes:
- Linear: Fully connected layer
- Layer_Norm: Layer Normalization
- Batch_Norm1D: 1D Batch Normalization
- Batch_Norm2D: 2D Batch Normalization
- DropOut: Dropout regularization
- Conv: 2D Convolutional layer
- Transpose_Conv: Transposed Convolutional layer
- Transpose_Conv_Rt: Reverse-type Transposed Convolutional layer
- Maxpool: Max pooling layer
- SELF_ATTENTION_HEAD: Self-attention mechanism
- MULTIHEAD_ATTENTION: Multi-head attention mechanism
- Convolution2D: Advance
- BLOCK: Single Transformer block
- SEQUENTIAL_BLOCKS: Sequence of Transformer blocks

Dependencies:
- numpy: For numerical operations
- SUPER_AVK.Utility_functions: For utility functions
- SUPER_AVK.AvK: For Tensor class
"""

import numpy as np
from .Utility_functions import normal_xavier, update_gradient, transpose_cov_out_shape, softmax
from .AvK import Tensor, concat

# Function to initialize parameters for neural network layers
def initialize_params(initialization, in_dims, out_dims, req_bias, require_grad) -> np.ndarray:
    """
    Initializes parameters for neural network layers.
    
    Args:
        initialization (str): Initialization method
        in_dims (int): Number of input dimensions
        out_dims (int): Number of output dimensions
        req_bias (bool): Whether to use bias
        require_grad (bool): Whether to compute gradients
    
    Returns:
        tuple: Tuple containing initialized weights and bias (if required)
    """
    if initialization == 'standard normal':
        Weights = Tensor(np.random.randn(in_dims, out_dims) * normal_xavier(in_dims, out_dims),\
                                _name='Weights', require_grad = require_grad)
        Bias = None
        if req_bias:
            Bias = Tensor(np.random.randn(1,out_dims) * normal_xavier(1,out_dims),\
                                _name='Bias', require_grad = require_grad)
            
    elif initialization == 'standard':
        Weights = Tensor(np.random.randn(in_dims, out_dims), _name='Weights', require_grad = require_grad)
        Bias = None
        if req_bias:
            Bias = Tensor(np.random.randn(1,out_dims), _name='Bias', require_grad = require_grad)
            
    elif initialization == 'uniform':
        Weights = Tensor(np.ones((in_dims, out_dims)) * normal_xavier(in_dims, out_dims),\
                                _name='Weights', require_grad = require_grad)
        Bias = None
        if req_bias:
            Bias = Tensor(np.zeros((1,out_dims)) * normal_xavier(in_dims, out_dims),\
                                _name='Bias', require_grad = require_grad)
    else:
        raise Exception('initialization method is invalid')
    
    return Weights, Bias

# Linear layer implementation
class Linear:
    """
    Implements a linear (fully connected) layer.
    
    Args:
        in_dims (int): Number of input dimensions
        out_dims (int): Number of output dimensions
        req_bias (bool): Whether to use bias
        nonliner (bool): Whether to apply non-linear activation
        act_func (str): Activation function to use
        require_grad (bool): Whether to compute gradients
        initialization (str): Weight initialization method
    """
    def __init__(self, in_dims: int, out_dims: int, req_bias: bool = False, nonliner = False,\
        act_func = 'relu', require_grad = True, initialization = 'standard normal'):
        self.req_bias = req_bias
        
        self.require_grad = require_grad
        self.nonliner = nonliner
        self.Weights, self.Bias = initialize_params(initialization, in_dims, out_dims, req_bias, require_grad)

        self.act_func = act_func
        
    def __call__(self, x):
        if self.req_bias:
            act = (x @ self.Weights) + self.Bias
        else:
            act = (x @ self.Weights)
        if self.nonliner:
            if self.act_func == 'relu':
                return act.relu()
            elif self.act_func == 'leaky relu':
                return act.leaky_relu()
            elif self.act_func == 'soft plus':
                return act.soft_plus()
            elif self.act_func == 'sigmoid':
                return act.sigmoid()
            else:
                raise Exception('wrong nonliner')
        else:
            return act
    
    def __repr__(self):
        return f'act func: {self.act_func}\nnon liner: {self.nonliner}\
            \nrequire grad: {self.require_grad}\
            \nWeights: {self.Weights.shape}\nBias: {self.Bias.shape}'
            

# Layer Normalization implementation
class Layer_Norm:
    """
    Implements Layer Normalization.
    
    Args:
        C (int): Number of channels/features
    """
    def __init__(self, C):
        self.gamma = Tensor(np.ones((1,C)) * normal_xavier(1, C), require_grad=True,\
                            _name='gamma')
        self.beta = Tensor(np.zeros((1,C)) * normal_xavier(1, C), require_grad=True,\
                          _name = 'beta')
    def __call__(self, x, ddof = 1):
        assert isinstance(x, Tensor), "only AvK.Tensor type for X(input)"
        NORM = x.layernorm(ddof)
        
        shift_scale = NORM * self.gamma + self.beta
        
        return shift_scale
    

# 1D Batch Normalization implementation
class Batch_Norm1D:
    """
    Implements 1D Batch Normalization for 2D or 3D Tensors.
    
    Args:
        C (int): Number of channels/features
        momentum (float): Momentum for running statistics
    """
    def __init__(self, C, momentum=0.001):
        """Batch Norm 1D work with 2D or 3D Tensor, with structure of (B, C) or (B, C, T)"""
        self.momentum = momentum
        self.gamma = Tensor(np.ones((1,C)), require_grad=True, _name='gamma') # * normal_xavier(1, C)
        self.beta = Tensor(np.zeros((1,C)), require_grad=True, _name = 'beta')
        
        self.running_var = Tensor(np.ones((1,C)), require_grad=False, _name = 'running var')
        self.running_mean = Tensor(np.zeros((1,C)), require_grad=False, _name = 'running mean')
        
    def __call__(self, x, mode= 'train', ddof = 1):
        assert isinstance(x, Tensor), "only AvK.Tensor type for X(input)"
        NORM = x.batchnorm1D(ddof)
        if mode == 'train':
            try:
                shift_scale = NORM * self.gamma + self.beta
            except:
                shift_scale = NORM * self.gamma.transpose(1,0) + self.beta.transpose(1,0)
                
            # Update running statistics
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.var(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.mean(axis=0, keepdims=True)
        elif mode == 'inferance':
            try:
                shift_scale = (x - self.running_mean) / (self.running_var + 1e-5)**0.5 * self.gamma + self.beta
            except:
                shift_scale = (x - self.running_mean.transpose(1,0)) / (self.running_var.transpose(1,0) + 1e-5)**0.5 * self.gamma.transpose(1,0) + self.beta.transpose(1,0)
        return shift_scale
    

# 2D Batch Normalization implementation
class Batch_Norm2D:
    """
    Implements 2D Batch Normalization for 4D Tensors.
    
    Args:
        C (int): Number of channels
        momentum (float): Momentum for running statistics
    """
    def __init__(self, C, momentum=0.001):
        """Batch Norm 2D work with 4D Tensor, with structure of (B, C, H, W)"""
        self.momentum = momentum
        self.gamma = Tensor(np.ones((1, C, 1, 1)), require_grad=True, _name='gamma') #* normal_xavier(1, C), 
        self.beta = Tensor(np.zeros((1, C, 1, 1)), require_grad=True, _name = 'beta')
        
        self.running_var = Tensor(np.ones((1, C, 1, 1)), require_grad=False, _name = 'running var')
        self.running_mean = Tensor(np.zeros((1, C, 1, 1)), require_grad=False, _name = 'running mean')
        
    def __call__(self, x, mode= 'train', ddof = 1):
        assert isinstance(x, Tensor), "only AvK.Tensor type for X(input)"
        if mode == 'train':
            NORM = x.batchnorm2D(ddof)
            shift_scale = NORM * self.gamma + self.beta
            
            # Update running statistics
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.var(axis=(0,2,3), keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.mean(axis=(0,2,3), keepdims=True)
        elif mode == 'inferance':
            NORM = (x - self.running_mean) / (self.running_var + 1e-5)**0.5
            shift_scale = NORM * self.gamma + self.beta
        
        return shift_scale

# Dropout layer implementation
class DropOut:
    """
    Implements Dropout regularization.
    
    Args:
        p (float): Dropout probability
    """
    def __init__(self, p = 0.5) -> None:
        self.p = p
        self.scale = 1/(1-self.p) if self.p < 1 else 1
    def __call__(self, X, drop_out = True):
        if drop_out:
            die_out_value = np.random.binomial(1, self.p, size = X.shape[1:])
            return X * self.scale * die_out_value 
        return X 
    
    def __repr__(self):
        return f'act func: {self.act_func}\nnon liner: {self.nonliner}\
            \nrequire grad: {self.require_grad}\
            \nWeights: {self.Weights.shape}\nBias: {self.Bias.shape}'
            
            
            
def mini_batches_GPT(data, max_sequence_len: int = 5, batch_size = 16):
    """
    Creates mini-batches for training a GPT model.

    Args:
        data (array-like): Input data for batching
        max_sequence_len (int): Maximum length of sequences
        batch_size (int): Number of samples in each batch

    Returns:
        tuple: Tuple of input and output mini-batches
    """
    pos = np.random.randint(low=0, high=len(data) - max_sequence_len, size=(batch_size, ))
    xs = np.array([data[i:i+max_sequence_len] for i in pos])
    ys = np.array([data[i + 1:i+max_sequence_len + 1] for i in pos])
    return xs, ys

def greedy_sampling(model_structure, prompts:list):
    """
    Performs greedy sampling from the model.

    Args:
        model_structure: The model structure to sample from
        prompts (list): List of input prompts

    Returns:
        array: Probabilities of the last token
    """
    logits = model_structure(prompts, drop_out = False)
    probs = np.float64(softmax(logits.data)[0])
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    return probs[-1]


def greedy_generator(model_structure, decode, T:int, prompts:list, max_new_tokens: int, num_samples: int = 1):
    """
    Generates sequences using a greedy approach.

    Args:
        model_structure: The model structure to generate from
        decode: Function to decode the generated tokens
        T (int): Length of the input sequence
        prompts (list): List of input prompts
        max_new_tokens (int): Maximum number of new tokens to generate
        num_samples (int): Number of samples to generate

    Returns:
        list: Updated list of prompts with generated tokens
    """
    for i in range(max_new_tokens):
        croped_prompts = np.array(prompts)[np.newaxis,:][:, -T:]
        probs = greedy_sampling(model_structure, croped_prompts)
        out = np.argmax(np.random.multinomial(num_samples, probs.ravel()))
        
        prompts.append(out)
    #     os.system('clear')
    #     print(decode(prompts))
    # os.system('clear')
    return prompts


class Beam_particle:
    """
    Represents a particle in beam search for sequence generation.

    Args:
        idx (int): Index of the current token
        prob (float): Probability of the current token
        value (list): List of tokens in the current sequence
        T (int): Length of the input sequence
        cumc_prob (float): Cumulative probability of the sequence
    """
    def __init__(self, idx, prob, value:list, T, cumc_prob=1) -> None:
        self.idx = idx
        self.prob = prob
        self.value = value
        self.T = T
        self.cumc_prob = cumc_prob
    def __call__(self, model_structure, beam_width):
        """
        Generates next set of beam particles.

        Args:
            model_structure: The model structure to generate from
            beam_width (int): Width of the beam search

        Returns:
            list: List of new Beam_particle objects
        """
        output = []
        next_prob = greedy_sampling(model_structure, np.array(self.value)[np.newaxis, :][:, -self.T:])
        next_idx = np.argsort(next_prob)[::-1][:beam_width]
        
        for i in range(beam_width):
            particle_idx = next_idx[i]
            particle_prob = next_prob[particle_idx]
            particle_value = self.value.copy()
            particle_value.append(particle_idx)
            beam_particle = Beam_particle(particle_idx, particle_prob, particle_value, self.T, cumc_prob= (self.prob*particle_prob) )

            output.append(beam_particle)
        return output
        
def beam_generator_wraper(model_structure, T:int, prompts:list, max_new_tokens: int, beam_width = 3):
    """
    Wrapper function for beam search generation.

    Args:
        model_structure: The model structure to generate from
        T (int): Length of the input sequence
        prompts (list): List of input prompts
        max_new_tokens (int): Maximum number of new tokens to generate
        beam_width (int): Width of the beam search

    Returns:
        list: Generated sequence of tokens
    """
    Pos1 = Beam_particle(prompts[-1], 1, prompts, T)(model_structure, beam_width)
    for _ in range(max_new_tokens):
        Beam_pool = [Pos1[i](model_structure, beam_width) for i in range(beam_width)]
        Beam_pool = [particle for particles in Beam_pool for particle in particles]
        Beam_pool.sort(key=lambda x: x.cumc_prob, reverse=True)
        Pos1 = Beam_pool[0:beam_width]

    Pos1.sort(key=lambda x: x.cumc_prob, reverse=True)
    return Pos1[0].value      
            
def getPositionEncoding(shape, n=100):
    """
    Generates positional encoding for transformer models.

    Args:
        shape (tuple): Shape of the input tensor
        n (int): Denominator term in the position encoding formula

    Returns:
        numpy.ndarray: Positional encoding matrix
    """
    P = np.zeros(shape[1::])
    seq_len = shape[1]
    d_model = shape[2]
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

class EMB_POS_encoding:
    def __init__(self, T,C, vocab):
        self.token_emb_w = Tensor(np.random.randn(vocab, C), require_grad=True)# * normal_xavier(T,C)  #(vocab,C)
    def __call__(self, X):
        token_emb = self.token_emb_w[X]
        emb = token_emb + Tensor(getPositionEncoding(token_emb.shape))
        
        return emb
    
class SELF_ATTENTION_HEAD:
    """
    Implements a single self-attention head for transformer models.

    Args:
        C (int): Number of input channels
        head_dims (int): Dimension of each attention head
    """
    def __init__(self, C, head_dims):
        self.C = C
        self.head_dims = head_dims
        
        self.query = Linear(C, head_dims,initialization='standard normal')
        self.key = Linear(C, head_dims,initialization='standard normal')
        self.value = Linear(C, head_dims,initialization='standard normal')
        
        self.drop_out = DropOut()
    def __call__(self, pos_emb, drop_out = True):
        """
        Computes self-attention for the given input.

        Args:
            pos_emb (Tensor): Input tensor with positional embeddings
            drop_out (bool): Whether to apply dropout

        Returns:
            Tensor: Output of the self-attention head
        """
        Q = self.query(pos_emb)
        K = self.key(pos_emb)
        V = self.value(pos_emb)

        attention = Q @ K.transpose(0,2,1)
        scale = attention * (self.C**-0.5)
        mask_in = scale.where(np.tril(scale.data) == 0, -np.inf, scale.data)

        mask_prob = mask_in.softmax()

        attention_values = (self.drop_out(mask_prob) @ V) if drop_out else (self.drop_out(mask_prob) @ V)
        
        return attention_values

        
class MULTIHEAD_ATTENTION:
    """
    Implements multi-head attention for transformer models.

    Args:
        n_heads (int): Number of attention heads
        C (int): Number of input channels
    """
    def __init__(self, n_heads,C):
        print(f'n heads: {n_heads}, C: {C}')
        assert C % n_heads == 0, f'{C} d_models cannot split into {n_heads} heads!'
        
        self.n_heads = n_heads
        self.head_dims = C//n_heads
        self.C = C

        self.proj = Linear(self.C, self.C, initialization='standard normal')
        self.heads = [SELF_ATTENTION_HEAD(self.C, self.head_dims) for _ in range(self.n_heads)]
        self.drop_out = DropOut()
    
    
    def __call__(self, pos_emb, drop_out = True):
        """
        Computes multi-head attention for the given input.

        Args:
            pos_emb (Tensor): Input tensor with positional embeddings
            drop_out (bool): Whether to apply dropout

        Returns:
            Tensor: Output of the multi-head attention
        """
        MH = concat([head(pos_emb) for head in self.heads]).transpose(1,2,0,3)
        rMH = self.drop_out(MH.reshape(*pos_emb.shape)) if drop_out else MH.reshape(*pos_emb.shape)
        
        proj_values = self.proj(rMH)
        return proj_values

class BLOCK:
    """
    Implements a single Transformer block.

    This class combines multi-head attention with feed-forward layers and layer normalization.
    It's a fundamental building block for Transformer-based architectures.

    Args:
        B (int): Batch size
        C (int): Number of channels/features
        n_heads (int): Number of attention heads

    Attributes:
        multihead_attention (MULTIHEAD_ATTENTION): Multi-head attention layer
        MH_norm (Layer_Norm): Layer normalization layer for multi-head attention
        ffw (Layer_Norm): Feed-forward layer
        ffw_norm (Layer_Norm): Layer normalization layer for feed-forward layer
    """
    def __init__(self, B,C, n_heads):
        self.multihead_attention = MULTIHEAD_ATTENTION(n_heads, C)
        # self.layer_norm = LAYER_NORM(C)
        
        self.MH_norm = Layer_Norm(C)
        self.ffw = Layer_Norm(C)
        self.ffw_norm = Layer_Norm(C)
        
    def __call__(self, x, drop_out = True):
        x = x + (self.multihead_attention(self.MH_norm(x), drop_out))
        x = x + self.ffw(self.ffw_norm(x), drop_out)
        return x
    
    
class SEQUENTIAL_BLOCKS:
    """
    Implements a sequence of Transformer blocks.

    This class applies a series of Transformer blocks to the input tensor.
    It's a fundamental building block for Transformer-based architectures.

    Args:
        B (int): Batch size
        C (int): Number of channels/features
        n_heads (int): Number of attention heads
        n_layers (int): Number of Transformer blocks in the sequence

    Attributes:
        n_layers (int): Number of Transformer blocks in the sequence
        blocks (list): List of Transformer blocks
    """
    def __init__(self, B, C, n_heads, n_layers):
        self.n_layers = n_layers
        
        self.blocks = [BLOCK(B, C, n_heads) for _ in range(n_layers)]
            
    def __call__(self, x, drop_out = True):
        """
        Applies the sequence of Transformer blocks to the input tensor.

        Args:
            x (Tensor): Input tensor
            drop_out (bool): Whether to apply dropout

        Returns:
            Tensor: Output of the sequence of Transformer blocks
        """
        for block in self.blocks:
            x = block(x, drop_out)
        return x
    
# Convolutional layer implementation
class Conv():
    """
    Implements a 2D convolutional layer.
    
    Args:
        X_dim (tuple): Input dimensions (depth, height, width)
        n_kernals (int): Number of kernels/filters
        h_kernal (int): Kernel height
        w_kernal (int): Kernel width
        stride (int): Stride of the convolution
        padding (int): Padding size
    """
    def __init__(self, X_dim, n_kernals, h_kernal, w_kernal, stride, padding):

        self.d_X, self.h_X, self.w_X = X_dim

        self.n_kernals, self.h_kernal, self.w_kernal = n_kernals, h_kernal, w_kernal
        self.stride, self.padding = stride, padding

        self.h_out = (self.h_X - h_kernal + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_kernal + 2 * padding) / stride + 1
    

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            print(f'{self.h_out=}')
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_shape = (self.n_kernals, self.h_out, self.w_out)

        self.W = Tensor(np.random.randn(
            n_kernals, self.d_X, h_kernal, w_kernal) , require_grad=True, _name = 'Kernal') #* normal_xavier(
            #      n_inp= self.d_X * self.h_X * self.w_X,
            #      n_outp= self.d_X * self.h_out * self.w_out
            #  )
            
        self.b = Tensor(np.random.randn(1, self.n_kernals, 1, 1), require_grad=True, _name = 'Bias')
        
        self.params = [self.W]
        
        
    def __call__(self, X):
        assert isinstance(X, Tensor), "only AvK.Tensor type for X(input)"
        X2col = X.im2col(self.h_kernal, self.w_kernal, self.stride, self.padding)
        size_col = self.h_kernal * self.w_kernal
        B = X.shape[0]
        
        X2col_batch = X2col.swapaxes(-1,-2).reshape(X.shape[0], 1, self.d_X, -1, size_col)
        K2row_batch = self.W.reshape(1, self.n_kernals, self.d_X, -1, 1)
        
        conv = (X2col_batch @ K2row_batch).sum(axis=2, keepdims=True).reshape(B, self.n_kernals, self.h_out, self.w_out) 
        
        return conv #+ self.b

# Transposed Convolutional layer implementation
class Transpose_Conv(): 
    """
    Implements a 2D transposed convolutional layer.
    
    Args:
        X_dim (tuple): Input dimensions
        original_shape (tuple): Original input shape before convolution
        n_kernals (int): Number of kernels/filters
        h_kernals (int): Kernel height
        w_kernals (int): Kernel width
        stride (int): Stride of the convolution
        padding (int): Padding size
    """
    def __init__(self, X_dim, original_shape, n_kernals, h_kernals, w_kernals, stride, padding):
        self.d_X, self.h_X, self.w_X = X_dim
        self.prev_dX, self.prev_hX, self.prev_wX = original_shape
        
        self.n_kernals = n_kernals 
        self.h_kernals = h_kernals
        self.w_kernals = w_kernals
        self.stride = stride
        self.padding = padding

        self.W = Tensor(np.random.randn(n_kernals, self.d_X, h_kernals, w_kernals), require_grad=True, _name = 'Kernal')## / np.sqrt(self.prev_n_Kernal / 2.)
        # self.b = Tensor(np.random.randn(1, self.prev_dX, 1, 1), require_grad=True, _name = 'Bias')
        self.params = [self.W]

        self.h_out = transpose_cov_out_shape(
            self.h_X, self.h_kernals, self.stride, self.padding)
        self.w_out = transpose_cov_out_shape(
            self.w_X, self.w_kernals, self.stride, self.padding)
    
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_shape = (self.n_kernals, self.h_out, self.w_out)


    def __call__(self, X):
        assert isinstance(X, Tensor), "only AvK.Tensor type for X(input)"
        prev_nX = X.shape[0]
        X2row_batch = X.reshape(prev_nX, 1, self.d_X, 1, -1)
        
        K2col_batch = self.W.reshape(1, self.n_kernals, self.d_X, -1, 1)#14h41 8/7 -> 17h31 10/7
        
        linear = (K2col_batch @ X2row_batch).sum(axis=2, keepdims=True)
        
        Tconv = linear.col2im((prev_nX, self.prev_dX, self.prev_hX, self.prev_wX), 
                              self.h_kernals, self.w_kernals, self.stride, self.padding)
        
        return Tconv #+ self.b

# Transposed Convolutional layer implementation (reverse-type)
class Transpose_Conv_Rt(): 
    """
    Implements a reverse-type 2D transposed convolutional layer.
    This layer takes parameters from the opposite convolution for upsampling purposes.
    
    Args:
        X_dim (tuple): Input dimensions
        prev_Conv (Conv): Previous convolutional layer
    """
    def __init__(self, X_dim, prev_Conv: Conv):
        '''Transpose convolution, reverse-type: mean that, this ones takes parameters fron the opposite convolution for upsampling purpose'''
        self.prev_dX, self.prev_hX, self.prev_wX = prev_Conv.d_X, prev_Conv.h_X, prev_Conv.w_X
        
        self.d_X, self.h_X, self.w_X = X_dim
        
        self.prev_n_Kernal = prev_Conv.n_kernals
        self.prev_h_kernal = prev_Conv.h_kernal
        self.prev_w_kernal = prev_Conv.w_kernal
        self.stride, self.padding = prev_Conv.stride, prev_Conv.padding
        
        self.n_kernals = self.prev_dX
        self.d_kernals = self.prev_n_Kernal
        self.h_kernals = self.prev_h_kernal
        self.w_kernals = self.prev_w_kernal
 
        self.W = Tensor(np.random.randn(
            self.n_kernals, self.d_kernals, self.h_kernals, self.w_kernals)
                        , require_grad=True, _name = 'Kernal')## / np.sqrt(self.prev_n_Kernal / 2.)
        
        # self.b = Tensor(np.random.randn(1, self.prev_dX, 1, 1), require_grad=True, _name = 'Bias')
        

        self.h_out = transpose_cov_out_shape(
            self.h_X, self.h_kernals, self.stride, self.padding)
        self.w_out = transpose_cov_out_shape(
            self.w_X, self.w_kernals, self.stride, self.padding)
    
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_shape = (self.n_kernals, self.h_out, self.w_out)

        
    def __call__(self, X):
        assert isinstance(X, Tensor), "only AvK.Tensor type for X(input)"
        prev_nX = X.shape[0]
        X2row_batch = X.reshape(prev_nX, 1, self.d_X, 1, -1)
        K2col_batch = self.W.reshape(1, self.n_kernals, self.d_X, -1, 1)#7h01 11/7
        
        linear = (K2col_batch @ X2row_batch).sum(axis=2, keepdims=True)
        
        Tconv = linear.col2im((
            prev_nX, self.prev_dX, self.prev_hX, self.prev_wX), 
                              self.prev_h_kernal, self.prev_w_kernal, self.stride, self.padding)
        
        return Tconv #+ self.b
        

# Maxpool layer implementation
class Maxpool():
    """
    Implements a 2D max pooling layer.
    
    Args:
        X_dim (tuple): Input dimensions
        size (int): Pooling window size
        stride (int): Stride of the pooling operation
    """
    def __init__(self, X_dim, size, stride=2):

        self.d_X, self.h_X, self.w_X = X_dim

        self.params = []

        self.size = size
        self.stride = stride

        self.h_out = (self.h_X - size) / stride + 1
        self.w_out = (self.w_X - size) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception(f"Invalid dimensions!\n\t{self.h_out=}\n\t{self.w_out=}")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_shape = (self.d_X, self.h_out, self.w_out)

    def __call__(self, X):
        assert isinstance(X, Tensor), "only AvK.Tensor type for X(input)"
        X2col = X.im2col(self.size, self.size, self.stride, padding=0)
        X2col_reshape = X2col.transpose(0,1,3,2).reshape(-1, self.size * self.size)
        
        max_indexes = X2col_reshape.argmax(axis=1)
        out = X2col_reshape[
            range(len(max_indexes.data)), max_indexes
                            ].reshape(-1, *self.out_shape)

        return out

# Convolution2D layer implementation
class Convolution2D:
    """
    Implements a 2D convolutional layer with optional pooling, batch normalization, and activation.
    
    Args:
        input_shape (tuple): Input shape
        n_kernals (int): Number of kernels
        kernel_size (int): Size of the kernels
        kernals_stride (int): Stride for convolution
        padding (int): Padding size
        pooling (bool): Whether to apply pooling
        pooling_kernals (int): Size of pooling kernels
        pool_stride (int): Stride for pooling
        nonliner (bool): Whether to apply non-linear activation
        act_func (str): Activation function to use
        batch_norm (bool): Whether to apply batch normalization
    """
    def __init__(self, input_shape: tuple, n_kernals: int, kernel_size: int, kernals_stride = 1,  padding = 0,\
        pooling = False, pooling_kernals = 2,  pool_stride = 2,\
        nonliner=False, act_func='relu',\
        batch_norm=True):
        
        self.CONV = Conv(input_shape, n_kernals, kernel_size, kernel_size, stride=kernals_stride, padding=padding)
        self.Pool = Maxpool(self.CONV.out_shape, pooling_kernals, pool_stride) if pooling else None
        if batch_norm and not pooling:
            self.BatchNorm = Batch_Norm2D(self.CONV.out_shape[0])
        elif batch_norm and pooling:
            self.BatchNorm = Batch_Norm2D(self.Pool.out_shape[0])
        
        self.nonliner = nonliner
        self.pooling = pooling
        
        self.act_func = act_func
        self.out_shape = self.Pool.out_shape if pooling else self.CONV.out_shape
        
        print(f'-{input_shape=}, {self.out_shape=}\
            \n\t+{n_kernals=}, {kernel_size=}, {kernals_stride=}, {padding=}\n\t+{pooling=}, {pooling_kernals=}, {pool_stride=}\n\t+{nonliner=}, {act_func=}\n\
        +{batch_norm=}\n')
        
    def __call__(self, x, mode = None):
        act = self.BatchNorm(self.Pool(self.CONV(x)), mode) if self.pooling else self.BatchNorm(self.CONV(x), mode)
        
        if self.nonliner:
            if self.act_func == 'relu':
                return act.relu()
            elif self.act_func == 'leaky relu':
                return act.leaky_relu()
            elif self.act_func == 'soft plus':
                return act.soft_plus()
            elif self.act_func == 'sigmoid':
                return act.sigmoid()
            else:
                raise Exception('wrong nonliner')
        else:
            return act
        

# Transpose Convolution2D layer implementation (reverse-type)
class Transpose_Convolution2D_RT:
    """
    Implements a reverse-type 2D transposed convolutional layer with optional upsampling,
    batch normalization, and activation.
    
    Args:
        input_shape (tuple): Input shape
        prev_CNN (Conv): Previous convolutional layer
        upsampling (bool): Whether to apply upsampling
        scale_factor (int): Scale factor for upsampling
        nonliner (bool): Whether to apply non-linear activation
        act_func (str): Activation function to use
        batch_norm (bool): Whether to apply batch normalization
    """
    def __init__(self, input_shape: tuple, prev_CNN: Conv,\
        upsampling = False, scale_factor=2, nonliner=False, act_func='relu',\
        batch_norm=False):
        
        self.nonliner = nonliner
        self.upsampling = upsampling
        
        self.UPSAMPLING = Upsampling(input_shape, scale_factor) if self.upsampling else None
        self.TCONV = Transpose_Conv_Rt(self.UPSAMPLING.out_shape, prev_CNN.CONV) if self.upsampling else Transpose_Conv_Rt(input_shape, prev_CNN.CONV)
        
        self.BatchNorm = Batch_Norm2D(self.TCONV.out_shape[0]) if batch_norm else None
        
        n_kernals = self.TCONV.n_kernals
        h_kernals = self.TCONV.h_kernals
        kernals_stride = self.TCONV.stride
        padding = self.TCONV.padding
        
        
        self.act_func = act_func
        self.out_shape = self.TCONV.out_shape
        
        
        print(f'-{input_shape=}, {self.out_shape=}\
            \n\t+{n_kernals=}, {h_kernals=}, {kernals_stride=}, {padding=}\n\t+{upsampling=}, {scale_factor=}\n\t+{nonliner=}, {act_func=}\n\t+{batch_norm=}\n')
        
    def __call__(self, x, mode):
        act = self.BatchNorm(self.TCONV(self.UPSAMPLING(x)), mode) if self.upsampling else self.BatchNorm(self.TCONV(x), mode)
        
        if self.nonliner:
            if self.act_func == 'relu':
                return act.relu()
            elif self.act_func == 'leaky relu':
                return act.leaky_relu()
            elif self.act_func == 'soft plus':
                return act.soft_plus()
            elif self.act_func == 'sigmoid':
                return act.sigmoid()
            else:
                raise Exception('wrong nonliner')
        else:
            return act
        
        

# Transpose Convolution2D layer implementation
class Transpose_Convolution2D:
    """
    Implements a 2D transposed convolutional layer with optional upsampling,
    batch normalization, and activation.
    
    Args:
        input_shape (tuple): Input shape
        target_shape (tuple): Target output shape
        n_kernals (int): Number of kernels
        h_kernals (int): Height of kernels
        w_kernals (int): Width of kernels
        stride (int): Stride for convolution
        padding (int): Padding size
        upsampling (bool): Whether to apply upsampling
        scale_factor (int): Scale factor for upsampling
        nonliner (bool): Whether to apply non-linear activation
        act_func (str): Activation function to use
        batch_norm (bool): Whether to apply batch normalization
    """
    def __init__(self, input_shape:tuple, target_shape:tuple, n_kernals, h_kernals, w_kernals, stride=1, padding=0,
        upsampling = False, scale_factor=2, nonliner=False, act_func='relu',
        batch_norm = False):
        
        self.nonliner = nonliner
        self.upsampling = upsampling
        
        self.UPSAMPLING = Upsampling(input_shape, scale_factor) if self.upsampling else None
        self.TCONV = Transpose_Conv(self.UPSAMPLING.out_shape, target_shape, n_kernals, h_kernals, w_kernals, stride, padding) if self.upsampling else Transpose_Conv(input_shape, target_shape, n_kernals, h_kernals, w_kernals, stride, padding)
        
        self.BatchNorm = Batch_Norm2D(self.TCONV.out_shape[0]) if batch_norm else None
                
        n_kernals = self.TCONV.n_kernals
        h_kernals = self.TCONV.h_kernals
        kernals_stride = self.TCONV.stride
        padding = self.TCONV.padding
        
        
        self.act_func = act_func
        self.out_shape = self.TCONV.out_shape
        
        
        print(f'-{input_shape=}, {self.out_shape=}\
            \n\t+{n_kernals=}, {h_kernals=}, {kernals_stride=}, {padding=}\n\t+{upsampling=}, {scale_factor=}\n\t+{nonliner=}, {act_func=}\n\t+{batch_norm=}\n')
        
    def __call__(self, x, mode):
        act = self.BatchNorm(self.TCONV(self.UPSAMPLING(x)), mode) if self.upsampling else self.BatchNorm(self.TCONV(x), mode)
        
        if self.nonliner:
            if self.act_func == 'relu':
                return act.relu()
            elif self.act_func == 'leaky relu':
                return act.leaky_relu()
            elif self.act_func == 'soft plus':
                return act.soft_plus()
            elif self.act_func == 'sigmoid':
                return act.sigmoid()
            else:
                raise Exception('wrong nonliner')
        else:
            return act
        
        

# Fractionally strided convolution implementation
class Fractionally_strided_convolution:
    """
    Placeholder for fractionally strided convolution implementation.
    """
    def __init__(self) -> None:
        pass

# Upsampling layer implementation
class Upsampling:
    """
    Implements an upsampling layer.
    
    Args:
        X_dim (tuple): Input dimensions
        scale_factor (int): Scale factor for upsampling
    """
    def __init__(self, X_dim, scale_factor = 2) -> None:
        self.C, self.H, self.W = X_dim
        self.C_out, self.H_out, self.W_out = self.C, self.H * scale_factor, self.W * scale_factor
        self.scale_factor = scale_factor
        
        self.out_shape = (self.C_out, self.H_out, self.W_out)
        
    def __call__(self, X):
        """
        Applies upsampling to the input tensor.

        Args:
            X (Tensor): Input tensor

        Returns:
            Tensor: Upsampled tensor
        """
        out = Tensor(np.repeat(X.data, self.scale_factor, axis=-2).repeat(self.scale_factor, axis=-1), (X,), 'up sampling')
        
        def _backward():
            dX = self.backward(out.grad)
            update_gradient(X, dX)
        out._backward = _backward
        return out
    
    def backward(self, glob_grad):
        """
        Computes the backward pass for the upsampling operation.

        Args:
            glob_grad (numpy.ndarray): Gradient from the next layer

        Returns:
            numpy.ndarray: Gradient with respect to the input
        """
        first_m = np.add.reduceat(glob_grad, np.mgrid[0:self.H_out: self.scale_factor], axis=-2)#rows
        second_m = np.add.reduceat(first_m, np.mgrid[0:self.W_out: self.scale_factor], axis=-1)#cols
        
        return second_m