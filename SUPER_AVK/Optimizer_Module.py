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

Dependencies:
- numpy: For numerical operations
- SUPER_AVK.Utility_functions: For utility functions
- SUPER_AVK.AvK: For Tensor class
"""

import numpy as np
from .AvK import Tensor

# Base class for optimization algorithms
class Optimization:
    """
    Base class for optimization algorithms.
    """
    def __init__(self, optimization_method, alpha,
                 lr, beta1, beta2, lamda):
        """
        Initialize the Optimization class.

        Args:
            optimization_method (str): The optimization method to use.
            alpha (float): Momentum factor.
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment estimate.
            beta2 (float): Exponential decay rate for second moment estimate.
            lamda (float): Weight decay factor.
        """
        self.optimization_method = optimization_method
        self._gradient_calculator = lambda: None
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamda = lamda
        
    def GD(self, node) -> np.ndarray:
        """
        Gradient Descent optimization step.

        Args:
            node (Tensor): The node to optimize.

        Returns:
            np.ndarray: The step to take in parameter space.
        """
        node.grad = np.round(node.grad, decimals=10)
        return node.grad * -self.lr
    
    def MGD(self, node, m) -> np.ndarray:
        """
        Momentum Gradient Descent optimization step.

        Args:
            node (Tensor): The node to optimize.
            m (np.ndarray): The momentum.

        Returns:
            np.ndarray: The step to take in parameter space.
        """
        node.grad = np.round(node.grad, decimals=10)
        return (node.grad * -self.lr) + (self.alpha * m) 

    def Adam(self, node, m, v) -> np.ndarray:
        """
        Adam optimization step.

        Args:
            node (Tensor): The node to optimize.
            m (np.ndarray): First moment vector.
            v (np.ndarray): Second moment vector.

        Returns:
            tuple: Updated first moment, second moment, and step to take.
        """
        node.grad = np.round(node.grad, decimals=10)
        first_m = (self.beta1 * np.round(m, decimals=10)) + (1 - self.beta1) * node.grad
        second_m = (self.beta2 * v) + (1 - self.beta2) * (node.grad)**2
        #Bias correction
        m_hat = first_m / (1 - self.beta1)
        v_hat = second_m / (1 - self.beta2)
        step = -self.lr * m_hat / (np.sqrt(v_hat)  + 1e-8)
        
        return first_m, second_m, step
    
    def AdamW(self, node, m, v) -> np.ndarray:
        """
        AdamW optimization step.

        Args:
            node (Tensor): The node to optimize.
            m (np.ndarray): First moment vector.
            v (np.ndarray): Second moment vector.

        Returns:
            tuple: Updated first moment, second moment, and step to take.
        """
        node.grad = np.round(node.grad, decimals=10)
        first_m = (self.beta1 * np.round(m, decimals=10)) + (1 - self.beta1) * node.grad
        second_m = (self.beta2 * v) + (1 - self.beta2) * (node.grad)**2
        # second_m = np.maximum(second_m, node._v)
        #Bias correction
        m_hat = first_m / (1 - self.beta1)
        v_hat = second_m / (1 - self.beta2)
        step = (-self.lr * m_hat / ((np.sqrt(v_hat)) + 1e-8) ) - self.lr*(self.lamda * node.data)
        
        return first_m, second_m, step
    
    
class Optimizer(Optimization):
    """
    Optimizer class that implements various optimization algorithms.
    """
    def __init__(self, optimization_method = 'GD', alpha = 0.5,
                 lr = 0.1, beta1 = 0.9, beta2 = 0.999, lamda = 0.01):
        """
        Initialize the Optimizer.

        Args:
            optimization_method (str): The optimization method to use.
            alpha (float): Momentum factor.
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment estimate.
            beta2 (float): Exponential decay rate for second moment estimate.
            lamda (float): Weight decay factor.
        """
        super().__init__(optimization_method, alpha, lr, beta1, beta2, lamda)
        self.backward_pass = lambda: None
        
        self._m = []
        self._v = []
        self.init_step = True
        
        if self.optimization_method == 'GD':
            self._gradient_calculator = self.GD
            self.backward_pass = self.GD_updater
        
        elif self.optimization_method == 'MGD':
            self._gradient_calculator = self.MGD
            self.backward_pass = self.MGD_updater
            
        elif self.optimization_method == 'Adam':
            self._gradient_calculator = self.Adam
            self.backward_pass = self.Adam_updater
        
        elif self.optimization_method == 'AdamW':
            self._gradient_calculator = self.AdamW
            self.backward_pass = self.AdamW_updater
        else:
            raise Exception('optimization_method is invalid')
    
    def MGD_updater(self, topo, retain_all_grad=False) -> None:
        """
        Update parameters using Momentum Gradient Descent.

        Args:
            topo (list): Topologically sorted list of nodes.
            retain_all_grad (bool): Whether to retain all gradients.
        """
        i = 0
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = pre_node.grad if (pre_node.retain_grad or retain_all_grad) else None
            if node.require_grad and self.init_step:
                step = self._gradient_calculator(node, 0)
                node.data += step
                
                self._m.append(step)

            elif node.require_grad and not self.init_step:
                step = self._gradient_calculator(node, self._m[i])

                node.data += step
                self._m[i] = step
                
                i += 1
            pre_node = node
        self.init_step = False
                
            
    def GD_updater(self, topo, retain_all_grad=False) -> None:
        """
        Update parameters using Gradient Descent.

        Args:
            topo (list): Topologically sorted list of nodes.
            retain_all_grad (bool): Whether to retain all gradients.
        """
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = pre_node.grad if (pre_node.retain_grad or retain_all_grad) else None
            if node.require_grad and self.init_step:
                step = self._gradient_calculator(node)
                node.data += step 

            elif node.require_grad and not self.init_step:
                step = self._gradient_calculator(node)
                node.data += step
                
            pre_node = node
        self.init_step = False
                
    def Adam_updater(self, topo, retain_all_grad=False) -> None:
        """
        Update parameters using Adam optimizer.

        Args:
            topo (list): Topologically sorted list of nodes.
            retain_all_grad (bool): Whether to retain all gradients.
        """
        i = 0
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = pre_node.grad if (pre_node.retain_grad or retain_all_grad) else None
            if node.require_grad and self.init_step:
                m, v, step = self._gradient_calculator(node, 0 ,0)
                node.data += step
                
                self._m.append(m)
                self._v.append(v)

            elif node.require_grad and not self.init_step:
                m, v, step = self._gradient_calculator(node, self._m[i], self._v[i])

                node.data += step
                self._m[i] = m
                self._v[i] = v

                i += 1
            pre_node = node
        self.init_step = False
                            
    def AdamW_updater(self, topo, retain_all_grad=False) -> None:
        """
        Update parameters using AdamW optimizer.

        Args:
            topo (list): Topologically sorted list of nodes.
            retain_all_grad (bool): Whether to retain all gradients.
        """
        i = 0
        pre_node = Tensor(0)
        for node in reversed(topo):
            node._backward()
            pre_node.grad = pre_node.grad if (pre_node.retain_grad or retain_all_grad) else None
            if node.require_grad and self.init_step:
                m, v, step = self._gradient_calculator(node, 0 ,0)
                node.data += step
                
                self._m.append(m)
                self._v.append(v)
                # print('ok1')
            
            elif node.require_grad and not self.init_step:
                m, v, step = self._gradient_calculator(node, self._m[i], self._v[i])

                node.data += step
                self._m[i] = m
                self._v[i] = v

                # print('ok2')
                i += 1
            # print('ok3')
            pre_node = node
        self.init_step = False
                         
                         
                         
    def zeros_grad(self, topo) -> None:
        """
        Set gradients of all nodes to zero.

        Args:
            topo (list): Topologically sorted list of nodes.
        """
        for node in reversed(topo):
            if node.shape != ():
                node.grad = np.zeros_like(node.data,dtype=node.dtype) #0
            else:
                node.grad = 0.0
                
    # def delete_topo(self, topo):
    #     for node in topo:
    #         del node
    
    def __repr__(self):
        """
        Return a string representation of the Optimizer.

        Returns:
            str: A string containing the optimizer's parameters.
        """
        return f'optimization_method: {self.optimization_method}\
            \nlr: {self.lr}\talpha: {self.alpha}\tbeta1: {self.beta1}\tbeta2: {self.beta2}\tlamda: {self.lamda}'