r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""

1.


A) The Jacobian tensor of the output will have the shape (N, out_features, in_features), in our case is (64, 514, 1024).<br />
B) W is considered sparse if it is initialized using a normal distribution with a low standard deviation.<br />
C)  Since this is a linear transformation, the Jacobian of the transformation with respect to X is W^T, and it doesn't need to be explicitly calculated.<br />
For the derivative with respect to W:<br />

2.


A) The shape of the partial derivative of the transformation with respect to W matches the shape of X, which is 64x512.<br />
B) X can be sparse depending on the input, but it isn't necessarily required for the model to learn efficiently, unlike W. <br />
C) The derivative with respect to W is simply X, as seen in the implementation of the linear layer, so there's no need to explicitly calculate the Jacobian.
"""

part1_q2 = r"""

Backpropagation is not the only method for training neural networks. It is the most commonly used method because, in most classification tasks, the output is a scalar, allowing gradients to be explicitly and efficiently calculated. This makes it straightforward to use the chain rule to compute the infinitesimal change in the loss function with respect to each parameter in the network.

In all models, there must be some objective function that is minimized, though not necessarily through backpropagation. For example, you can have an MLP where the last layer is trained using Least Squares optimization. The previous layers can be initialized randomly, and learning can be achieved by pruning unnecessary connections within the second-to-last layer.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 0.1, 0.00001    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.1
    reg = 0.00001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr_vanilla = 5e-2
    lr_momentum = 1e-3
    lr_rmsprop = 1e-4
    reg = 5e-5
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-2
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

The graphs align with our expectations.
With a high dropout rate of 0.8, the training accuracy is lower compared to the scenario with no dropout, as neural networks trained with lower dropout rates tend to overfit the data.

Since dropout is a regularization technique, we anticipate that the neural network will perform better on the test dataset when regularization is applied with a higher probability. Consequently, in the test accuracy results, the neural network with a high dropout rate shows higher accuracy due to its improved generalization.

"""

part2_q2 = r"""

Both cross-entropy loss and accuracy can simultaneously increase during the testing phase. Accuracy measures the ratio of correctly classified samples by selecting the maximum argument in the probability vector induced by cross-entropy. The loss is calculated using the formula $-y^t \log(\hat{y})$, which is equivalent to $-x_i + \log(e^{x_1} + ... + e^{x_n})$, where $i$ is the correct output label.

This scenario can occur if the entropy increases while the correct label retains the highest probability. For instance, consider a mini-batch with two samples. The cross-entropy loss might decrease for one sample but increase more for the other, leading to an overall increase in the average cross-entropy loss, even though the classification accuracy improves.

For example, in binary classification, suppose in the first epoch we have the following softmax activations:
1. $[0.1, 0.9]$ with the correct label $[0, 1]$
2. $[0.49, 0.51]$ with the correct label $[1, 0]$

In the second epoch, the activations change to:
1. $[0.4, 0.6]$ with the correct label $[0, 1]$
2. $[0.51, 0.49]$ with the correct label $[1, 0]$

The accuracy for the first mini-batch is 50%, and for the second mini-batch, it is 100%, showing an increase.

The loss for the first mini-batch is:
$
\left( -0.9 + \log(e^{0.1} + e^{0.9}) - 0.49 + \log(e^{0.49} + e^{0.51}) \right) / 2 = 0.5371 
$

The loss for the second mini-batch is:
$
\left( -0.6 + \log(e^{0.4} + e^{0.6}) - 0.51 + \log(e^{0.49} + e^{0.51}) \right) / 2 = 0.64 
$

Thus, both loss and accuracy increase.

In the initial epochs, the classifier might achieve higher accuracy, but the cross-entropy loss can increase, leading to an overall increase in loss. This issue typically resolves after a few batches as the cross-entropy between the model's predictions and the true distribution decreases.

"""

part2_q3 = r"""

1) Backpropagation is a method for training neural networks by using the chain rule to calculate the derivative of the loss function with respect to parameters like weights and biases. Gradient descent is a general method for finding a local minimum of a function by taking steps in the direction of the gradient, which points to the steepest ascent. Negating the gradient points to the steepest descent. The assumption is that updating parameters in the direction of steepest descent of the loss function will lead to a local, or often global, minimum of the loss function.

2) In gradient descent, the gradient of the loss is calculated using the entire training dataset, requiring \(N\) calculations, whereas stochastic gradient descent (SGD) samples uniformly from the dataset, requiring only one calculation per iteration. With SGD, convergence isn't guaranteed in every iteration but over many iterations, the expected value approximates the true gradient. In gradient descent, the gradient closely resembles the expected value in every iteration due to using \(N\) samples. When far from the minima, SGD converges more rapidly than gradient descent but stays at a higher loss near the minima because updates are based on randomly picked samples, never fully capturing the optimal direction. SGD's convergence rate is sublinear $((O(1/k)))$, while gradient descent has a linear rate $((O(c^k)))$.

3) To achieve a norm of less than $(\epsilon > 0)$ between the $(k)$-th iteration parameters and the minimizer, SGD requires $(O(1/\epsilon))$ resources, independent of $(N)$. Gradient descent requires $(O(N \log(1/\epsilon)))$ resources, dependent on $(N)$. Given large datasets and limited computational resources, SGD is often more practical. Additionally, SGD quickly minimizes loss compared to gradient descent, even though its asymptotic loss is slightly higher.

4) In SGD, the distance between the minima and the parameters in the \(k\)-th iteration decreases sublinearly, i.e., \(O(1/k)\). In gradient descent, the rate of convergence is linear, i.e., \(O(c^k)\).


To achieve a norm of less than \(\epsilon > 0\) in Euclidean norm between the \(k\)-th iteration parameters and the minimizer, SGD requires \(O(1/\epsilon)\) computational resources, which is independent of \(N\) (training set size). Gradient descent requires \(O(N \log(1/\epsilon))\) resources, which is dependent on \(N\). In real-world settings, datasets are usually very large and computational resources are limited, making SGD more relevant. Additionally, SGD minimizes loss more quickly compared to gradient descent, even though the asymptotic loss is slightly higher with SGD.

When training with gradient descent during backpropagation, weights are updated by multiplying \(d_{\text{out}} * W^T\), where \(d_{\text{out}}\) has dimensions \((d_{\text{out}}, N)\) and \(W\) has dimensions \((d_{\text{in}}, N)\). In the described approach, the change in \(W\) is calculated as the sum of matrices resulting from multiplying \((d_{\text{out}}, \text{batch size})\) by \((\text{batch size}, d_{\text{in}})\). Algebraically, these operations are distinct.

We also need to cache inputs for each linear layer of each batch, leading to memory usage similar to gradient descent but done sequentially. Eventually, saving the inputs for each layer of each batch will exceed the device's memory capacity. Since the device cannot handle full gradient descent, incrementally increasing memory usage to match gradient descent will cause the device to run out of memory, as explained by the intermediate value theorem.


"""

part2_q4 = r"""


1) 



A) To compute the gradient $\nabla f(x_0)$ using forward mode automatic differentiation (AD) with reduced memory complexity, follow these steps:

Given $f = f_n \circ f_{n-1} \circ \ldots \circ f_1$:

1. **Initialization**:
   - Let $y_0 = x_0$.
   - Initialize the derivative $\dot{y}_0 = 1$ (this represents $\frac{dy_0}{dx_0}$).

2. **Propagation through each $f_i$**:
   - For $i = 1, 2, \ldots, n$:
     1. Compute $y_i = f_i(y_{i-1})$.
     2. Compute the derivative $\dot{y}_i$ using:
        \[
        \dot{y}_i = f_i'(y_{i-1}) \cdot \dot{y}_{i-1}
        \]

3. **Final Output**:
   - The value of $f(x_0)$ is $y_n$.
   - The gradient $\nabla f(x_0)$ is $\dot{y}_n$.


- **Function values**: Store only the current $y_i$, leading to $\mathcal{O}(1)$ space.
- **Derivative values**: Store only the current $\dot{y}_i$, also $\mathcal{O}(1)$ space.


The memory complexity for computing the gradient using forward mode AD is:
$
\boxed{\mathcal{O}(1)}
$

This approach maintains the $\mathcal{O}(n)$ computation cost while achieving efficient memory usage.

B) To compute the gradient $\nabla f(x_0)$ using backward mode automatic differentiation (AD) with reduced memory complexity, follow these steps:


Given $f = f_n \circ f_{n-1} \circ \ldots \circ f_1$:

1. **Forward Pass**:
   - Compute the values of each intermediate variable:
     \[
     y_i = f_i(y_{i-1}) \quad \text{for} \quad i = 1, 2, \ldots, n
     \]
   - Store the values $y_0, y_1, \ldots, y_n$ for use in the backward pass.

2. **Backward Pass**:
   - Initialize the adjoint (reverse mode derivative) for the final output:
     \[
     \bar{y}_n = 1
     \]
   - Propagate adjoints backward through each $f_i$:
     \[
     \bar{y}_{i-1} = \bar{y}_i \cdot f_i'(y_{i-1}) \quad \text{for} \quad i = n, n-1, \ldots, 1
     \]
   - The gradient $\nabla f(x_0)$ is $\bar{y}_0$.

In backward mode AD, we have to store all intermediate values $y_0, y_1, \ldots, y_n$ because the backward pass requires these values for gradient computation. Therefore, the memory complexity is dominated by the storage of these intermediate values.

- **Intermediate values**: Storing $y_0, y_1, \ldots, y_n$ requires $\mathcal{O}(n)$ space.
- **Adjoints**: Storing the adjoints $\bar{y}_i$ during the backward pass also requires $\mathcal{O}(n)$ space, but they can be reused since only the current and previous adjoints are needed at any step.

The memory complexity for computing the gradient using backward mode AD is:
\[
\boxed{\mathcal{O}(n)}
\]

This approach maintains the $\mathcal{O}(n)$ computation cost, but requires $\mathcal{O}(n)$ memory to store intermediate values and adjoints.


2) Yes, these techniques can be generalized for arbitrary computational graphs. Forward mode AD propagates derivatives through each node in sequence, and backward mode AD computes adjoints from the output node back to the input nodes. Both methods manage memory efficiently by storing only necessary intermediate values and derivatives.



3) In deep architectures like VGGs and ResNets, backward mode AD (backpropagation) benefits significantly. By storing intermediate values during the forward pass and reusing them during the backward pass, memory usage is optimized. This allows efficient gradient computation, crucial for training deep networks with numerous layers and parameters.







"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    activation = 'relu'
    hidden_dims = 10
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.001
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""

1) Our training accuracy remains relatively high up to a certain threshold, which limits its peak value. The optimization error could be improved by adjusting the model's hyperparameters or changing the optimizer. Modifying these aspects could result in a better optimization error.

2) There is only a small difference between the test set loss and the training set loss. This indicates that the model is not overfitting to the training set, resulting in good generalization and a low generalization error.

3) High approximation error is mainly indicated by a high loss. Our training loss is relatively high, suggesting a significant approximation error. To reduce this error, we could enhance the model by increasing the number of channels.

"""

part3_q2 = r"""

When evaluating the validation set, we anticipate a higher False Negative Rate (FNR) compared to the False Positive Rate (FPR). This expectation arises from the comparison between a positive number and a probability, suggesting a higher likelihood of observing a higher FNR on the validation set. However, this assumption is not always accurate, as our observations indicate instances where the FPR can be equal to or even higher than the FNR.

"""

part3_q3 = r"""

1) Our objective is to elevate the False Positive Rate (FPR) in relation to the False Negative Rate (FNR). This can be accomplished by raising the threshold to minimize the overall cost. By minimizing the cost, we effectively decrease the FNR, subsequently reducing the risk of failing to diagnose patients with non-lethal symptoms.

2) Conversely, in this scenario, our aim is to amplify the False Negative Rate (FNR) in comparison to the False Positive Rate (FPR). One approach to achieve this is by lowering the threshold for diagnosing patients with ambiguous symptoms that have a high likelihood of being fatal.

"""


part3_q4 = r"""

1) From the results that we obtained, we can see that for a model with a depth of one, the decision boundary is very close to being linear. Once we played with the values of the model's width and kept depth=1 we managed to reach more complex decision boundaries.

We also noticed that when we increased the depth, the values of the width gained more importance. Once we increased the depth we also got more complex decision boundaries. The increase in depth also yielded models with better performance on the validation set.

2)Our results show that when width is increasing we get a better validation accuracy , and the threshold is being updated due to changing of the FPR and FNR.

3)For depth = 1 and width = 32 : valid_acc = 93% and test_acc = 84.0%

4) The validation set serves multiple purposes: it is employed to fine-tune hyperparameters and optimize the threshold value. This fine-tuning process contributes to achieving higher accuracy on the test set. Consequently, the hyperparameters selected based on the validation set aid in generalizing our model and ultimately optimizing test set accuracy.


"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.2
    loss_fn = torch.nn.CrossEntropyLoss()
    momentum = 0.003
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
A standard block that performs two convolutions with kernels on an input channel of 256 has parameters. In contrast, the bottleneck block has parameters due to additional techniques that reduce the parameter count. The bottleneck block uses a convolution for channel dimension projection, followed by a convolution (similar to the standard block) to extract spatial information, and concludes with another convolution to restore the original 256 channels. This method significantly reduces the parameter count.

The bottleneck block also requires fewer floating-point operations compared to the regular block because it processes the input through fewer convolutions, thereby reducing computational cost. For an input size of , the number of floating-point operations can be calculated. The regular block requires operations, whereas the bottleneck block requires operations.

Additionally, the bottleneck block affects both the spatial dimensions and the number of feature maps of the input, unlike the regular block which only impacts the input spatially. By projecting the channels to a lower dimension and then restoring them to the original dimension, the bottleneck block enables the network to capture spatial and cross-feature map information more effectively, despite having fewer parameters. This approach reduces computational complexity while maintaining the network's representation capability. Overall, the bottleneck block significantly reduces both parameters and complexity compared to the regular block.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""