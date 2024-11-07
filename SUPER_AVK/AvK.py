"""
Tensor Class Documentation
===========================

The `Tensor` class provides a comprehensive set of functionalities for tensor manipulation and automatic differentiation. Below is a list of essential functions available in the `Tensor` class:

Initialization and Representation:
----------------------------------
- `__init__(self, data, _prev=(), _op='', _name='None', dtype=np.float32, require_grad=False, retain_grad=False)`: Initialize a Tensor object.
- `__repr__(self)`: Return a string representation of the Tensor.

Data Export and Loading:
------------------------
- `data_export(self, path='./')`: Export tensor data to a file.
- `load_data(self, path='./')`: Load tensor data from a file.

Graph and Parameter Management:
-------------------------------
- `forward_graph(self, file_name: str)`: Generate a forward computation graph and save it to a file.
- `n_parameters(self, optimizer=None)`: Count the number of parameters in the model.
- `backward(self, optimizer, update=False, zeros_grad=True, retain_all_grad=False)`: Perform backpropagation through the computation graph.
- `detach(self)`: Detach the tensor from its computation history.
- `copy(self)`: Create a copy of the tensor.

Basic Arithmetic Operations:
----------------------------
- `__neg__(self)`: Negate the tensor.
- `__add__(self, other)`: Add two tensors.
- `__radd__(self, other)`: Add two tensors (reversed).
- `__sub__(self, other)`: Subtract two tensors.
- `__rsub__(self, other)`: Subtract two tensors (reversed).
- `__mul__(self, other)`: Multiply two tensors.
- `__rmul__(self, other)`: Multiply two tensors (reversed).
- `__truediv__(self, other)`: Divide two tensors.
- `__rtruediv__(self, other)`: Divide two tensors (reversed).
- `__pow__(self, other)`: Raise the tensor to a power.
- `__matmul__(self, other)`: Perform matrix multiplication.

Array Operations:
-----------------
- `__getitem__(self, idx)`: Get an item from the tensor.
- `max(self, axis=-1, keepdims=False)`: Compute the maximum value along an axis.
- `sum(self, axis=None, keepdims=False)`: Compute the sum of the tensor elements.
- `argmax(self, axis=None, keepdims=False)`: Compute the indices of the maximum values along an axis.

Shape Operations:
-----------------
- `transpose(self, *axes)`: Transpose the tensor.
- `reshape(self, *new_shape)`: Reshape the tensor.
- `swapaxes(self, axis1, axis2)`: Swap two axes of the tensor.
- `squeeze(self)`: Remove single-dimensional entries from the shape of the tensor.
- `flatten(self)`: Flatten the tensor.

Activation Functions:
---------------------
- `exp(self, eps=0)`: Compute the exponential of the tensor.
- `log(self)`: Compute the natural logarithm of the tensor.
- `mean(self, axis=None, keepdims=True)`: Compute the mean of the tensor elements.
- `var(self, axis=None, ddof=0, keepdims=True)`: Compute the variance of the tensor elements.
- `std(self, axis=None, ddof=1, keepdims=True)`: Compute the standard deviation of the tensor elements.
- `layernorm(self, ddof)`: Apply layer normalization.
- `batchnorm1D(self, ddof)`: Apply 1D batch normalization.
- `batchnorm2D(self, ddof=0)`: Apply 2D batch normalization.
- `where(self, condition, if_True, if_False)`: Apply element-wise conditional operation.
- `tril(self, k=0)`: Get the lower triangular part of the tensor.
- `relu(self)`: Apply the ReLU activation function.
- `leaky_relu(self, alpha=0.01)`: Apply the Leaky ReLU activation function.
- `soft_plus(self)`: Apply the soft plus activation function.
- `sigmoid(self)`: Apply the sigmoid activation function.
- `softmax(self, axis=-1)`: Apply the softmax function.
- `log_softmax(self, axis=-1)`: Apply the log softmax function.

Image Processing Functions:
---------------------------
- `im2col(self, kernal_height, kernal_width, stride, padding)`: Convert image data to column format for convolution operations.
- `col2im(self, x_shape, kernal_height, kernal_width, stride, padding)`: Convert column format back to image data.

Loss Functions:
----------------
- `cross_entropy_loss(self, target)`: Compute the cross-entropy loss.
- `MSE(self, target)`: Compute the Mean Squared Error loss.
- `KL_divergence(self, target)`: Compute the Kullback-Leibler divergence.
- `ELBO(self, mean, log_var, target, Beta=1)`: Compute the Evidence Lower Bound (ELBO) for variational autoencoders.
- `VQ_VAE_loss(self, encoder_output, codebook_vector, target, Beta=1)`: Compute the loss for a Vector Quantized Variational Autoencoder (VQ-VAE).

Miscellaneous Functions:
------------------------
- `concat(Tensor_list: list)`: Concatenate a list of tensors.
"""

import os
import gc
import numpy as np
from .Utility_functions import type_checking, Adam_parameters_init, update_gradient,\
    softmax, log_softmax, reverse_transpose, im2col_indices, col2im_indices

class Tensor:
    """
    Tensor Class
    ============

    The `Tensor` class provides a comprehensive set of functionalities for tensor manipulation 
    and automatic differentiation. It includes:

    - Initialization and representation
    - Data export and loading
    - Graph and parameter management
    - Basic arithmetic operations
    - Array operations
    - Shape operations
    - Activation functions
    - Image processing functions
    - Loss functions

    For detailed information on specific methods, please refer to their individual docstrings.
    """
    def __init__(self, data, _prev=(), _op='', _name='None', dtype=np.float32, require_grad=False, retain_grad=False, is_parameters = True):
        """
        Initialize a Tensor object.

        Args:
            data: The data to initialize the tensor with.
            _prev (tuple): Previous tensors in the computation graph.
            _op (str): The operation that created this tensor.
            _name (str): Name of the tensor.
            dtype (np.dtype): Data type of the tensor.
            require_grad (bool): Whether to compute gradients for this tensor.
            retain_grad (bool): Whether to retain gradients after backpropagation.
        """
        self.data, self.grad = type_checking(data, dtype)

        self.dtype = dtype
        self._backward = lambda: None
        self._prev = _prev
        
        self._op = _op
        self._name = _name
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        self.require_grad = require_grad
        self.retain_grad = retain_grad
        self.is_parameters = is_parameters
        
        self.topo = None
        Adam_parameters_init(self, self.require_grad, self.dtype)

    def data_export(self, path='./'):
        """
        Export tensor data to a file.

        Args:
            path (str): The path to save the exported data.
        """
        DATA_SET = []
        #build forward topo
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        #reverse for backward topo, if req_grad then add data to DATA_SET
        for node in reversed(topo):
            if node.require_grad:
                DATA_SET.append(node.data)
        
        # directory = os.path.split(path)[0]
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        #     print(f"The new directory named {directory} is created!\n")
        np.savez(path, *DATA_SET)
        
        print(f'DATA SET EXPORT SUCCESS! -> path: {path}\n')
        
        
    def load_data(self, path='./'):
        """
        Load tensor data from a file.

        Args:
            path (str): The path to load the data from.
        """
        print(path)
        try:
            DATA_SET = np.load(path, allow_pickle=True) 
        except:
            print('LOAD FAILED! -> DATA SET directory path is not exist!\n')
            return 
        print(f'DATA SET LOAD SUCCESS! -> DATA SET is passing through computation tree!\npath: {path}\n')
        #build forward topo
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        #reverse for backward topo, if req_grad then set node data to correct set
        i = 0
        for node in reversed(topo):
            if node.require_grad:
                node.data = DATA_SET[f'arr_{i}']
                i+=1
                
                
    def __repr__(self):
        """
        Return a string representation of the Tensor.

        Returns:
            str: A string representation of the Tensor.
        """
        return f'Shape: {self.shape}\tdtype: {self.data.dtype}\trequire grad: {self.require_grad}\t_op: {self._op}\t_name: {self._name}\
        \nData:\n {self.data}'
    
    def forward_graph(self, file_name:str):
        """
        Generate a forward computation graph and save it to a file.

        Args:
            file_name (str): The name of the file to save the graph.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        try: 
            os.remove(f"{file_name}.txt")
        except:
            pass
        f = open(f"{file_name}.txt", "a")
        for node in topo:
            _name = node._name if node._name != 'None' else ''
            _op = node._op if node._op != '' else 'INP'
            
            if node.shape != ():
                f.write(f'[{_op}] {node.shape} {_name} ->  ')
            else:
                f.write(f'[{_op}]scalar ({node.data}) {_name} ->  ')
        f.close()
        
    def n_parameters(self, optimizer = None):
        """
        Count the number of parameters in the model.

        Args:
            optimizer: The optimizer to update with the parameter count.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        num_parameters = 0
        for node in reversed(topo):
            # node._backward()
            if node.require_grad:
                num_parameters += node.data.size
        print(f'numbers of parameters: {num_parameters:_}')
        
        if not optimizer == None:
            optimizer.num_parameters = num_parameters
           
    def backward(self, optimizer, update=False, zeros_grad=True, retain_all_grad = False):
        """
        Perform backpropagation through the computation graph.

        Args:
            optimizer: The optimizer to use for gradient updates.
            update (bool): Whether to update gradients.
            zeros_grad (bool): Whether to zero out gradients before backpropagation.
            retain_all_grad (bool): Whether to retain all gradients.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.topo = topo
        if zeros_grad:
            optimizer.zeros_grad(self.topo)

        if update:
            self.grad = np.ones_like(self.data)
            optimizer.backward_pass(self.topo, retain_all_grad)

        else:
            self.grad = np.ones_like(self.data)
            for node in reversed(self.topo):
                node._backward()
    
    def clean_tensor(self):
        del self.data
        del self.grad
        
    def clean_neural_net(self):
        for node in self.topo:
            node.clean_tensor() if not node.is_parameters else None #del if is_parameters == False
        gc.collect()           
        
    def copy(self):
        return Tensor(self.data, ())

    # Basic arithmetic operations
    def __neg__(self): # -self
        return self * -1


    def __add__(self, other):
        """Add two tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, (self, other), '+', is_parameters=False)
        
        def _backward():
            update_gradient(self, output.grad)
            update_gradient(other, output.grad)
            
        output._backward = _backward
        return output
    
    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other - self

    def __mul__(self, other):
        """Multiply two tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data, (self, other), '*', is_parameters=False)   
        def _backward():
            update_gradient(self, (other.data * output.grad))
            update_gradient(other, (self.data * output.grad))
                
        output._backward = _backward
        return output
    
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        try:
            return self * (other**-1)
        except:
            return self * (np.float32(other)**-1)

    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Tensor(self.data**other, (self,), f'**{other}', is_parameters=False)
        def _backward():
            update_gradient(self, (other * (self.data ** (other - 1)) * output.grad))
        output._backward = _backward
        return output
    
    def __matmul__(self, other):
        """Perform matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data @ other.data, (self, other), '@', is_parameters=False)

        def _backward():
            dA = output.grad @ np.moveaxis(other.data, -1, -2)
            dB = np.moveaxis(self.data, -1, -2) @ output.grad
            
            distA = dA.ndim - self.grad.ndim
            distB = dB.ndim - other.grad.ndim
            
            dA = dA.sum(axis=tuple(np.arange(distA)))
            dB = dB.sum(axis=tuple(np.arange(distB)))
            
            diffA = np.abs(np.array(self.shape) - np.array(dA.shape))
            self.grad += dA.sum(axis=tuple(np.where(diffA >= 1)[0]), keepdims=True)
            
            diffB = np.abs(np.array(other.shape) - np.array(dB.shape))
            other.grad += dB.sum(axis=tuple(np.where(diffB >= 1)[0]), keepdims=True)
            
        output._backward = _backward
        return output
    
    # Array operations
    def __getitem__(self, idx):
        # if type(idx) == slice or type(idx) == np.int_ or type(idx) == int or type(idx) == np.float32 or len(idx) == 1:
        #     output = Tensor(self.data[idx], (self, ), 'index')
        
        if isinstance(idx, (slice, int, float)) or len(idx) == 1 or (isinstance(idx, np.ndarray)):
            output = Tensor(self.data[idx], (self, ), 'index', is_parameters=False)
        
        elif len(idx) == 2:
            list_idx = list(idx)
            empty_array = []
            for i in range(len(list_idx)):
                try:
                    empty_array.append(np.int32(list_idx[i]))
                except:
                    empty_array.append(np.int32(list_idx[i].data))
            output = Tensor(self.data[empty_array[0], empty_array[1]], (self, ), 'index', is_parameters=False)
            
        def _backward():
            capsule = np.zeros_like(self.data)
            if type(idx) == slice or type(idx) == np.ndarray: 
                capsule[idx] = output.grad
            elif len(idx) == 1:
                capsule[idx] = output.grad
            elif len(idx) == 2:
                capsule[empty_array[0], empty_array[1]] = output.grad
            update_gradient(self, capsule)
        output._backward = _backward
        return output

    
    def max(self, axis = -1, keepdims = False):
        """Compute the maximum value along an axis."""
        act_data = np.max(self.data, axis=axis, keepdims=keepdims)
        output = Tensor(act_data, (self,), f'max', is_parameters=False)
        def _backward():
            update_gradient(self, np.where(self.data == act_data, 1, 0) * output.grad)
        output._backward = _backward
        return output
    
    
    def sum(self, axis = None, keepdims = False):
        act_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        output = Tensor(act_data, (self,), f'sum', is_parameters=False)
        def _backward():
            update_gradient(self, (np.ones_like(act_data) * output.grad))
        output._backward = _backward
        return output
    
    def argmax(self, axis = None, keepdims = False):
        assert len(self.shape) <= 2, 'Only supported 2D array for argmax!'
        act_data = np.argmax(self.data, axis=axis, keepdims=keepdims)
        output = Tensor(act_data, (self,), f'argmax', is_parameters=False)
        def _backward():
            capsule = np.zeros_like(self.data)
            if axis == 0:
                capsule[range(self.shape[1]), act_data] = output.grad
            elif axis == 1:
                capsule[range(self.shape[0]), act_data] = output.grad    
            else:
                raise Exception('EROOR!')
              
            update_gradient(self, capsule)
        output._backward = _backward
        return output
    
    # Shape operations
    def transpose(self, *axes):
        output = Tensor(self.data.transpose(axes), (self,), f'T', is_parameters=False)
        def _backward():
            update_gradient(self, reverse_transpose(output.grad, axes))
        output._backward = _backward
        return output
    
    def reshape(self, *new_shape):
        """Reshape the tensor."""
        output = Tensor(self.data.reshape(new_shape), (self,), f'reshape', is_parameters=False)
        def _backward():
            update_gradient(self, output.grad.reshape(self.shape))
        output._backward = _backward
        return output
    
    def swapaxes(self, axis1, axis2):
        output = Tensor(self.data.swapaxes(axis1, axis2), (self,), f'swapaxes', is_parameters=False)
        def _backward():
            update_gradient(self, output.grad.swapaxes(axis2, axis1))
        output._backward = _backward
        return output
    
    def squeeze(self):
        output = Tensor(np.squeeze(self.data), (self,), f'squeeze', is_parameters=False)
        def _backward():
            update_gradient(self, output.grad.reshape(self.shape))
        output._backward = _backward
        return output
    
    def flatten(self):
        output = Tensor(self.data.flatten(), (self,), f'flatten', is_parameters=False)
        def _backward():
            update_gradient(self, output.grad.reshape(self.shape))
        output._backward = _backward
        return output

    # Activation functions
    def exp(self, eps = 0):
        output = Tensor(np.exp(self.data + eps), (self,), f'exp', is_parameters=False)
        def _backward():
            update_gradient(self, output.data * output.grad)
        output._backward = _backward
        return output
    
    
    def log(self):
        output = Tensor(np.log(self.data), (self,), f'log', is_parameters=False)
        def _backward():
            update_gradient(self, (self.data)**-1 * output.grad)
        output._backward = _backward
        return output
    
    def mean(self, axis=None, keepdims=True):
        if axis == None:
            axis = tuple(range(0, self.data.ndim))
            
        sum = self.sum(axis, keepdims)
        try:
            N = np.prod([self.shape[i] for i in axis])
        except:
            N = self.shape[axis]
        mean = sum / N####mean = sum / len(self.data[..., 0][0])
        return mean

    def var(self, axis=None, ddof=0, keepdims=True):
        if axis == None:
            axis = tuple(range(0, self.data.ndim))
            
        mean = self.mean(axis)
        MSE = ((self - mean)**2).sum(axis, keepdims)
        try:
            N = np.prod([self.shape[i] for i in axis]) - ddof
        except:
            N = self.shape[axis] - ddof
        var_output = (MSE/ N)
        return var_output
    
    def std(self, axis=None, ddof=1, keepdims=True):
        if axis == None:
            axis = tuple(range(0, self.data.ndim))
        var = self.var(axis, ddof, keepdims) 
        std_output = (var)**0.5
        return std_output

    def layernorm(self, ddof):
        x = self
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, ddof = ddof, keepdims=True)
        NORM = (x - mean) / (std + 1e-5)
        return NORM
    
    def batchnorm1D(self, ddof):
        assert len(self.shape) == 2 or len(self.shape) == 3, "You used [BatchNorm1D] but input shape is not 2D or 3D"
        x = self
        if len(self.shape) == 2:
            axis = 0
        else: 
            axis = (0,2)
        mean = x.mean(axis, keepdims=True)
        std = x.std(axis, ddof = ddof, keepdims=True)
        NORM = (x - mean) / (std + 1e-5)
        return NORM
    
    
    def batchnorm2D(self, ddof = 0):
        assert len(self.shape) == 4,"You used [BatchNorm2D] but input shape is not 4D"
        x = self
        axis = (0,2,3)
        mean = x.mean(axis, keepdims=True)
        std = x.std(axis, ddof = ddof, keepdims=True)
        NORM = (x - mean) / (std + 1e-5)
        return NORM
    
    
    def where(self, condition, if_True, if_False):
        output = Tensor(np.where(condition, if_True, if_False), (self,), 'where', is_parameters=False)
        
        def _backward():
            update_gradient(self, np.where(condition, 0, 1) * output.grad)
        output._backward = _backward
        return output
    
    def tril(self, k = 0):
        output = Tensor(np.tril(self.data, k), (self,), 'tril', is_parameters=False)
        
        def _backward():
           update_gradient(self, np.where(output.data == 0, 0, 1) * output.grad)
        output._backward = _backward
        return output
    
    def relu(self):
        """Apply the ReLU activation function."""
        output = Tensor(np.where(self.data >=0, self.data, 0), (self,), 'ReLU', is_parameters=False)
        def _backward():
            update_gradient(self, np.where(output.data >0, 1, 0) * output.grad)
        output._backward = _backward

        return output
    
    def leaky_relu(self, alpha = 0.01):
        output = Tensor(np.where(self.data < 0, self.data * alpha, self.data), (self,), 'Leaky_ReLU', is_parameters=False)
        def _backward():
            update_gradient(self, np.where(output.data > 0, 1, alpha) * output.grad)
        output._backward = _backward
        return output
    
    def soft_plus(self):
        output = Tensor(np.log(1+np.exp(self.data)), (self,), 'Soft plus', is_parameters=False)
        def _backward():
            update_gradient(self, (1 / (1 + np.exp(-self.data))) * output.grad)
        output._backward = _backward
        return output
    
    def sigmoid(self):
        """Apply the sigmoid activation function."""
        output = Tensor(1 / (1+np.exp(-self.data)), (self,), 'Sigmoid', is_parameters=False)
        def _backward():
            update_gradient(self, (output.data * (1-output.data)) * output.grad)
        output._backward = _backward
        return output
    
    def softmax(self, axis=-1):
        """Apply the softmax function."""
        self -= self.max(axis, keepdims=True)
        exp = self.exp()
        probs = exp / (exp.sum(axis, keepdims=True)) 
        probs._op = 'softmax'
        return probs
    
    def log_softmax(self, axis=-1): 
        """
        Apply the log softmax function.

        Args:
            axis (int): The axis along which to compute the log softmax.

        Returns:
            Tensor: The result of the log softmax operation.
        """
        self -= self.max(axis, keepdims=True)
        
        if len(self.shape) == 2:
            out = self - self.exp().sum(axis).log().reshape(self.shape[0],  -1)
        elif len(self.shape) == 3:
            out = self - self.exp().sum(axis).log().reshape(self.shape[1],  -1)
        else:
            raise Exception(f'Invalid log_softmax operation on {self}')
        
        return out
    
    # Image processing functions
    def im2col(self, kernal_height, kernal_width, stride, padding):
        """Convert image data to column format for convolution operations."""
        output = Tensor( im2col_indices(self.data, kernal_height, kernal_width, stride, padding), (self,), 'Im2Col', is_parameters=False)
        def _backward():
            update_gradient(self, ( col2im_indices(output.grad, self.shape, kernal_height, kernal_width, stride, padding)))
        output._backward = _backward
        return output

    def col2im(self, x_shape, kernal_height, kernal_width, stride, padding):
        output = Tensor( col2im_indices(self.data, x_shape, kernal_height, kernal_width, stride, padding), (self,), 'Col2Im', is_parameters=False)
        def _backward():
            update_gradient(self, ( im2col_indices(output.grad, kernal_height, kernal_width, stride, padding).reshape(*self.shape)))
        output._backward = _backward
        return output
        
    # Loss functions
    def cross_entropy_loss(self, target):
        """
        Compute the cross-entropy loss.

        Args:
            target: The target values.

        Returns:
            Tensor: The computed cross-entropy loss.
        """
        x = self.data.copy()
        if len(x.shape) == 4:
            raise Exception(f'4d array, not supported!')
        elif len(x.shape) == 3:
            B = x.shape[0]
            T = x.shape[1]
        elif len(x.shape) == 2:
            B = x.shape[0]
            T = 1
        else:
            raise Exception('cross entropy error', f'data shape {self.shape}')
        target =  np.array(target)
        probs = softmax(x)
        logprobs = np.vstack(log_softmax(self))
        loss = -logprobs[range(B*T), target.flatten()].mean()

        output = Tensor(loss, _prev=(self,), _op = f'cross entropy loss', is_parameters=False)

        def _backward():
            dlogits = np.vstack(probs)
            dlogits[range(B*T), target.flatten()] -= 1
            dlogits = dlogits.reshape(probs.shape)
            
            self.grad = dlogits

        output._backward = _backward
        return output
    
    def MSE(self, target):
        """
        Compute the Mean Squared Error loss.

        Args:
            target: The target values.

        Returns:
            Tensor: The computed MSE loss.
        """
        assert self.data.size == target.data.size, 'x and target have different size( MSE require x and target to have the same size)'

        return ((self.flatten() - target.flatten())**2).sum(axis=(0)) / (self.data.size)
    
    def KL_divergence(self, target):
        """
        Compute the Kullback-Leibler divergence.

        Args:
            target: The target distribution.

        Returns:
            Tensor: The computed KL divergence.
        """
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        assert self.data.size == target.data.size, 'x and target have different size( KL_divergence require x and target to have the same size)'
        
        probs = self.softmax()
        
        KL_D = ((target / probs).log() * target)
        N = (np.sum([i for i in probs.shape]))
        AXIS = tuple([i for i in range(probs.data.ndim)])

        return KL_D.sum(axis=AXIS) / N
    
    def ELBO(self, mean, log_var, target, Beta = 1):
        """
        Compute the Evidence Lower Bound (ELBO) for variational autoencoders.

        Args:
            mean: The mean of the latent distribution.
            log_var: The log variance of the latent distribution.
            target: The target values.
            Beta (float): The beta parameter for KL divergence weighting.

        Returns:
            tuple: MSE, KL divergence, and negative ELBO.
        """
        assert isinstance(mean, Tensor), "only AvK.Tensor type for mean"
        assert isinstance(log_var, Tensor), "only AvK.Tensor type for log_var"
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        assert self.data.size == target.data.size, 'x and target have different size( ELBO require x and target to have the same size)'
        
        MSE = self.MSE(target)
        KL_divergence_closed_form =  mean**2 + log_var.exp(eps=1e-5) - 1 - log_var
        KL_divergence_closed_form = KL_divergence_closed_form.sum() / (2) #negtive

        ELBO = -MSE + (Beta * -KL_divergence_closed_form)
        NELBO = -ELBO
        
        return MSE, KL_divergence_closed_form, NELBO
    
    def VQ_VAE_loss(self, encoder_output, codebook_vector, target, Beta = 1):
        """
        Compute the loss for a Vector Quantized Variational Autoencoder (VQ-VAE).

        Args:
            encoder_output: The output of the encoder.
            codebook_vector: The codebook vector.
            target: The target values.
            Beta (float): The beta parameter for commitment loss weighting.

        Returns:
            tuple: RMSE, alignment loss, commitment loss, and total loss.
        """
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        
        RMSE = self.MSE(target)
        ALIGNMENT_LOSS = encoder_output.MSE(codebook_vector.copy())
        COMMITMENT_LOSS = Beta * codebook_vector.MSE(encoder_output.copy())
        
        LOSS = RMSE + ALIGNMENT_LOSS + COMMITMENT_LOSS
        return RMSE, ALIGNMENT_LOSS, COMMITMENT_LOSS, LOSS
        

    def concat(Tensor_list: list):
        """
        Concatenate a list of tensors.

        Args:
            Tensor_list (list): A list of Tensor objects to concatenate.

        Returns:
            Tensor: The concatenated tensor.
        """
        act_data = np.array([Tensor.data for Tensor in Tensor_list])
        output = Tensor(act_data, (tuple(Tensor_list)), _op = f'cat', is_parameters=False)
        def _backward():
            for tensor, i in zip(Tensor_list, range(len(Tensor_list))):
                tensor.grad += np.ones_like(tensor.data) * output.grad[i] #(nh, B, T, n_dim)
        output._backward = _backward
        return output

    def KL_divergence(self, target):
        """
        Compute the Kullback-Leibler divergence.

        Args:
            target: The target distribution.

        Returns:
            Tensor: The computed KL divergence.
        """
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        assert self.data.size == target.data.size, 'x and target have different size( KL_divergence require x and target to have the same size)'
        
        probs = self.softmax()
        
        KL_D = ((target / probs).log() * target)
        N = (np.sum([i for i in probs.shape]))
        AXIS = tuple([i for i in range(probs.data.ndim)])

        return KL_D.sum(axis=AXIS) / N
    
    def ELBO(self, mean, log_var, target, Beta = 1):
        """
        Compute the Evidence Lower Bound (ELBO) for variational autoencoders.

        Args:
            mean: The mean of the latent distribution.
            log_var: The log variance of the latent distribution.
            target: The target values.
            Beta (float): The beta parameter for KL divergence weighting.

        Returns:
            tuple: MSE, KL divergence, and negative ELBO.
        """
        assert isinstance(mean, Tensor), "only AvK.Tensor type for mean"
        assert isinstance(log_var, Tensor), "only AvK.Tensor type for log_var"
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        assert self.data.size == target.data.size, 'x and target have different size( ELBO require x and target to have the same size)'
        
        MSE = self.MSE(target)
        KL_divergence_closed_form =  mean**2 + log_var.exp(eps=1e-5) - 1 - log_var
        KL_divergence_closed_form = KL_divergence_closed_form.sum() / (2) #negtive

        ELBO = -MSE + (Beta * -KL_divergence_closed_form)
        NELBO = -ELBO
        
        return MSE, KL_divergence_closed_form, NELBO
    
    def VQ_VAE_loss(self, encoder_output, codebook_vector, target, Beta = 1):
        """
        Compute the loss for a Vector Quantized Variational Autoencoder (VQ-VAE).

        Args:
            encoder_output: The output of the encoder.
            codebook_vector: The codebook vector.
            target: The target values.
            Beta (float): The beta parameter for commitment loss weighting.

        Returns:
            tuple: RMSE, alignment loss, commitment loss, and total loss.
        """
        assert isinstance(target, Tensor), "only AvK.Tensor type for target"
        
        RMSE = self.MSE(target)
        ALIGNMENT_LOSS = encoder_output.MSE(codebook_vector.copy())
        COMMITMENT_LOSS = Beta * codebook_vector.MSE(encoder_output.copy())
        
        LOSS = RMSE + ALIGNMENT_LOSS + COMMITMENT_LOSS
        return RMSE, ALIGNMENT_LOSS, COMMITMENT_LOSS, LOSS
        

def concat(Tensor_list: list):
    """
    Concatenate a list of tensors.

    Args:
        Tensor_list (list): A list of Tensor objects to concatenate.

    Returns:
        Tensor: The concatenated tensor.
    """
    act_data = np.array([Tensor.data for Tensor in Tensor_list])
    output = Tensor(act_data, (tuple(Tensor_list)), _op = f'cat', is_parameters=False)
    def _backward():
        for tensor, i in zip(Tensor_list, range(len(Tensor_list))):
            tensor.grad += np.ones_like(tensor.data) * output.grad[i] #(nh, B, T, n_dim)
    output._backward = _backward
    return output
