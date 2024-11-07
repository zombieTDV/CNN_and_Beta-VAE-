"""
Utility Functions for Neural Network Operations

This module contains various utility functions used in neural network operations,
including parameter initialization, gradient handling, activation functions,
and shape calculations for different layer types.

Functions:
    - lr_scheduler: Adjust learning rate based on epoch number
    - Adam_parameters_init: Initialize Adam optimizer parameters
    - reverse_broardcast: Reverse broadcasting effect on gradients
    - update_gradient: Update node gradients, handling broadcasting
    - normal_xavier: Calculate Xavier normal initialization factor
    - reverse_transpose: Reverse the effect of a transpose
    - type_checking: Check and convert data types, create gradient arrays
    - softmax: Compute softmax probabilities
    - log_softmax: Compute log softmax values
    - get_im2col_indices: Get indices for im2col operation
    - im2col_indices: Perform im2col operation using fancy indexing
    - col2im_indices: Perform col2im operation using fancy indexing
    - cov_out_shape: Calculate output shape of convolution operation
    - transpose_cov_out_shape: Calculate output shape of transpose convolution
    - maxpool_out_shape: Calculate output shape of max pooling operation
    - up_sampling_out_shape: Calculate output shape of upsampling operation
    - up_cov_shape_solver_for_Pad: Calculate padding for upsampling + convolution
    - up_cov_shape_solver_for_Stride: Calculate stride for upsampling + convolution
"""

import numpy as np
import gc

def lr_scheduler(optimizer, epoch, initial_lr=1e-3, decay_rate=0.9, decay_step=10):
    """
    Adjust learning rate based on epoch number using step decay.

    Args:
        optimizer: The optimizer whose learning rate will be adjusted
        epoch (int): Current epoch number
        initial_lr (float): Initial learning rate. Default is 1e-3
        decay_rate (float): Factor by which learning rate is reduced. Default is 0.9
        decay_step (int): Number of epochs before each decay. Default is 10

    Returns:
        optimizer: The optimizer with updated learning rate

    Example:
        If initial_lr=0.001, decay_rate=0.9, decay_step=10:
        - Epochs 0-9: lr = 0.001
        - Epochs 10-19: lr = 0.0009
        - Epochs 20-29: lr = 0.00081
        And so on...
    """
    # Calculate new learning rate using step decay formula
    lr = initial_lr * (decay_rate ** (epoch // decay_step))
    # Update optimizer's learning rate
    optimizer.lr = lr
    return optimizer

def Adam_parameters_init(node, require_grad: bool, dtype: np.ndarray) -> None:
    """
    Initialize Adam optimizer parameters for a given node.

    Args:
        node: The node to initialize parameters for.
        require_grad (bool): Whether gradients are required for this node.
        dtype (np.ndarray): The data type for the initialized parameters.

    Returns:
        None
    """
    if require_grad:
        node._m = np.zeros_like(node.grad, dtype)
        node._v = node._m.copy()
    return None

def reverse_broardcast(node, grad: np.ndarray):
    """
    Reverse the broadcasting effect on gradients.

    Args:
        node: The node to reverse broadcast for.
        grad (np.ndarray): The gradient to reverse broadcast.

    Returns:
        np.ndarray: The reversed broadcasted gradient.
    """
    # Reduce dimensions if grad has more dimensions than node
    while grad.ndim > node.ndim:
        grad = grad.sum(axis=0)         
    
    # Sum along axes where node has singleton dimensions
    for i, k in zip(range(len(node.shape)), node.shape):
        if k == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def update_gradient(node, grad: np.ndarray) -> None:
    """
    Update the gradient of a node, handling broadcasting if necessary.

    Args:
        node: The node to update the gradient for.
        grad (np.ndarray): The gradient to add.

    Returns:
        None
    """
    try:
        node.grad += grad  # Try normal addition
    except: 
        # If broadcasting is needed, reverse it and then add
        grad = reverse_broardcast(node, grad)
        node.grad += grad

def normal_xavier(n_inp, n_outp) -> float:
    """
    Calculate the Xavier normal initialization factor.

    Args:
        n_inp: Number of input units.
        n_outp: Number of output units.

    Returns:
        float: The Xavier normal initialization factor.
    """
    return np.sqrt(2 / (n_inp + n_outp))

def reverse_transpose(x: np.ndarray, axes):
    """
    Reverse the effect of a transpose operation.

    Args:
        x (np.ndarray): The array to reverse transpose.
        axes: The axes used in the original transpose.

    Returns:
        np.ndarray: The reverse transposed array.
    """
    index_changed = {O: T for O, T in zip(range(len(axes)), axes)}
    
    origin_index = {T: O for O, T in index_changed.items()}
    origin_index = dict(sorted(origin_index.items()))
    origin_index = tuple(origin_index.values())

    return x.transpose(origin_index)

def type_checking(data, dtype) -> np.ndarray:
    """
    Check and convert data to the specified dtype, and create a gradient array.

    Args:
        data: The input data.
        dtype: The desired data type.

    Returns:
        tuple: (converted_data, gradient)
    """
    if isinstance(data, dtype):
        data = data
    else:
        data = np.array(data, dtype=dtype)
    if data.shape != ():
        grad = np.zeros_like(data, dtype=dtype)
    else:
        grad = 0   
    return data, grad

def softmax(x, axis=-1):
    """
    Compute the softmax of the input array.

    Args:
        x: Input array.
        axis (int): Axis along which to compute softmax. Default is -1.

    Returns:
        np.ndarray: Softmax probabilities.
    """
    x -= np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    probs = exp / (np.sum(exp, axis, keepdims=True)) 
    return probs
        
def log_softmax(input, axis=-1): 
    """
    Compute the log softmax of the input array.

    Args:
        input: Input array.
        axis (int): Axis along which to compute log softmax. Default is -1.

    Returns:
        np.ndarray: Log softmax values.

    Raises:
        Exception: If the input shape is not 2D or 3D.
    """
    data = input.data.copy()
    data -= np.max(data, axis, keepdims=True)
    
    if len(data.shape) == 2:
        out = data - np.log(np.sum(np.exp(data), axis)).reshape(data.shape[0],  -1)
    elif len(data.shape) == 3:
        out = data - np.log(np.sum(np.exp(data), axis)).reshape(data.shape[0], data.shape[1], -1)
    else:
        raise Exception(f'Invalid log_softmax operation on {input}')
    
    return out

def get_im2col_indices(x_shape, kernal_height, kernal_width, stride, padding):
    """
    Get indices for im2col operation.

    Args:
        x_shape: Shape of the input array.
        kernal_height (int): Height of the kernel.
        kernal_width (int): Width of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding size.

    Returns:
        tuple: (k, i, j) indices for im2col operation.
    """
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - kernal_height) % stride == 0
    assert (W + 2 * padding - kernal_height) % stride == 0
    out_height = np.int8((H + 2 * padding - kernal_height) / stride + 1)
    out_width = np.int8((W + 2 * padding - kernal_width) / stride + 1)
#   if not ((H + 2 * padding - kernal_height) / stride + 1).is_integer():
#       raise Exception('There is some problem with out height')

    i0 = np.repeat(np.arange(kernal_height), kernal_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(kernal_width), kernal_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernal_height * kernal_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, kernal_height, kernal_width, stride, padding):
    """
    Perform im2col operation using fancy indexing.

    Args:
        x (np.ndarray): Input array.
        kernal_height (int): Height of the kernel.
        kernal_width (int): Width of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding size.

    Returns:
        np.ndarray: Reshaped columns.
    """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, kernal_height, kernal_width, stride,
                                 padding)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    B = x.shape[0]
    size = kernal_height * kernal_width
    cols = cols.reshape(B, C, size, -1)
    return cols

def col2im_indices(cols, x_shape, kernal_height, kernal_width, stride, padding):
    """
    Perform col2im operation using fancy indexing.

    Args:
        cols (np.ndarray): Input columns.
        x_shape: Shape of the original input.
        kernal_height (int): Height of the kernel.
        kernal_width (int): Width of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding size.

    Returns:
        np.ndarray: Reconstructed array.
    """
    N, C, H, W = x_shape #that wierd, it sould be B, C, H, W but it seem to get the correct answer but idk why...
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, kernal_height, kernal_width, stride, padding)
    cols_reshaped = cols.reshape(N, C * kernal_height * kernal_width, -1)
#   cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def cov_out_shape(x: int, kernal: int, stride: int, padding: int):
    """
    Calculate the output shape of a convolution operation.

    Args:
        x (int): Input size.
        kernal (int): Kernel size.
        stride (int): Stride of the convolution.
        padding (int): Padding size.

    Returns:
        float: Output size of the convolution.
    """
    return ((x + 2 * padding - 1 * (kernal - 1) - 1) / stride + 1)

def transpose_cov_out_shape(x: int, kernal: int, stride: int, padding: int):
    """
    Calculate the output shape of a transpose convolution operation.

    Args:
        x (int): Input size.
        kernal (int): Kernel size.
        stride (int): Stride of the convolution.
        padding (int): Padding size.

    Returns:
        float: Output size of the transpose convolution.
    """
    return ((x - 1) * stride - 2 * padding + 1 * (kernal - 1) + 1)

def maxpool_out_shape(x: int, kernal: int, stride: int, padding: int):
    """
    Calculate the output shape of a max pooling operation.

    Args:
        x (int): Input size.
        kernal (int): Kernel size.
        stride (int): Stride of the pooling.
        padding (int): Padding size.

    Returns:
        float: Output size of the max pooling.
    """
    return ((x + 2 * padding - 1 * (kernal - 1) - 1) / stride + 1)

def up_sampling_out_shape(x: int, scale_factor: int):
    """
    Calculate the output shape of an upsampling operation.

    Args:
        x (int): Input size.
        scale_factor (int): Scale factor for upsampling.

    Returns:
        int: Output size of the upsampling.
    """
    return (x * scale_factor)

def up_cov_shape_solver_for_Pad(x: int, y: int, scale_factor: int, kernal: int, stride: int):
    """
    Calculate the padding required for upsampling followed by convolution to match a target shape.

    Args:
        x (int): Current shape.
        y (int): Target shape.
        scale_factor (int): Scale factor for upsampling.
        kernal (int): Kernel size.
        stride (int): Stride of the convolution.

    Returns:
        float: Required padding.
    """
    new_pad = ((stride * y - stride - x * scale_factor + 1 * (kernal - 1) + 1) / 2)
    return new_pad

def up_cov_shape_solver_for_Stride(x: int, y: int, scale_factor: int, kernal: int, padding: int):
    """
    Calculate the stride required for upsampling followed by convolution to match a target shape.

    Args:
        x (int): Current shape.
        y (int): Target shape.
        scale_factor (int): Scale factor for upsampling.
        kernal (int): Kernel size.
        padding (int): Padding size.

    Returns:
        float: Required stride.
    """
    new_stride = (x * scale_factor + 2 * padding - kernal) / (y - 1) 
    return new_stride
