r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**1**

**1.1 Full Jacobian Shape:**

The total number of elements in $\mat{X}$ is $N \times D_{\text{in}} = 64 \times 1024 = 65,536$.

The total number of elements in $\mat{Y}$ is $N \times D_{\text{out}} = 64 \times 512 = 32,768$.

The shape of the full Jacobian $\frac{\partial \mat{Y}}{\partial \mat{X}}$ is:

$$(N \times D_{\text{out}}) \times (N \times D_{\text{in}}) = (32,768) \times (65,536) = 2,147,483,648$$

**1.2 Block Matrix Structure:**

When viewed as a block matrix with blocks of shape $(D_{\text{out}} \times D_{\text{in}}) = (512 \times 1024)$:

The Jacobian is a **diagonal block matrix**. The structure is:
- **Diagonal blocks** (where $i = j$): Each diagonal block equals $\mat{W}^T$ because sample $i$'s output depends only on sample $i$'s input
- **Off-diagonal blocks** (where $i \neq j$): All zeros, because the output of sample $i$ is independent of the input of sample $j$

**1.3 Optimization:**

Because the Jacobian is a diagonal block matrix whose diagonal blocks are all identical (each block equals $\mat{W}^\top$), we do not need to store every block separately. Instead, we can store a single copy of the weight matrix.

- **Optimized storage:** one copy of the weight matrix $\mat{W}$ (or $\mat{W}^\top$) with shape $(512) \times (1024)$.

This reduces storage from the full Jacobian size of $(32,768)\times(65,536)$ elements down to $512\times1024 = 524{,}288$ elements.

**1.4 Computing Gradient Without Materializing Jacobian:**
Given $\delta\mathbf{Y}\in\mathbb{R}^{N\times512}$, compute per-sample
$$\delta\mathbf{x}^{(i)} = \delta\mathbf{y}^{(i)}\,\mathbf{W}\quad((1,512)\cdot(512,1024)=(1,1024))$$

Vectorized:

$$
\delta X = \begin{bmatrix}
\delta x^{(1)} \\
\delta x^{(2)} \\
\vdots \\
\delta x^{(N)}
\end{bmatrix}
= \begin{bmatrix}
\delta y^{(1)} \\
\delta y^{(2)} \\
\vdots \\
\delta y^{(N)}
\end{bmatrix} W
= \delta Y\,W
$$

Pass the matrix $\delta X$ to the previous layer's backward as its downstream gradient to continue propagation.

**1.5 Jacobian w.r.t. Weights:**
For the Jacobian $\frac{\partial \mat{Y}}{\partial \mat{W}}$:

- **Full Jacobian shape:**

$$
(N\cdot D_{\mathrm{out}}) \times (D_{\mathrm{out}}\cdot D_{\mathrm{in}}) = (32{,}768) \times (524{,}288) = 17,179,869,184.
$$

- **Block shape (if arranged as blocks):**

$$
(D_{\mathrm{out}}\times D_{\mathrm{in}}) = (512\times 1024).
$$

Brief explanation: For a single sample $i$ and output unit $p$ we have
$y_{i,p}=\sum_r W_{p,r}x_{i,r}$, so differentiating w.r.t. the $p$-th row of
$W$ yields the input row $x^{(i)}$. Thus each output's derivative places
the vector $x^{(i)}$ in the corresponding output-row, producing blocks of
size $D_{\mathrm{out}}\times D_{\mathrm{in}}$.
"""

part1_q2 = r"""
**Second-Order Derivatives in Gradient Descent**

The second-order derivative (Hessian) can be helpful in optimization, but it depends on the context:

**Why NOT always helpful:**

1. **Computational Cost**: Computing the full Hessian matrix is extremely expensive. For a network with $n$ parameters, the Hessian is $n \times n$, requiring $\mathcal{O}(n^2)$ memory and $\mathcal{O}(n^3)$ time for eigendecomposition.

2. **Storage**: For modern deep networks with millions of parameters, storing the full Hessian is infeasible.

**When Second-Order Derivatives ARE Helpful:**

   **Newton's Method**: Uses the inverse Hessian to determine step direction:
   $$\vec{\theta}_{t+1} = \vec{\theta}_t - \mathcal{H}^{-1} \nabla L$$
   Mult by $\mathcal{H}^{-1}$ automatically adjusts the step size, taking larger steps in flat regions (low curvature) and smaller steps in steep regions (high curvature), based on the function's second derivative.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.08, 0
    
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1,
        0.05,
        0.008,
        0.001,
        0.001,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    
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
        0.1,
        0.003,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======

    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
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
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 2 # number of layers (not including output)
    hidden_dims = 500  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    
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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.001, 1e-4, 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 (CNN) answers

def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    # ====== YOUR CODE: ======
    # CrossEntropyLoss is ideal for multiclass classification.
    # It combines LogSoftmax and NLLLoss in one single class.
    # Crucially, it expects raw logits as input (which our CNN provides).
    loss_fn = torch.nn.CrossEntropyLoss()

    # Hyperparameters for SGD:
    # lr: 0.01 is a safe starting point for many CNN architectures.
    # weight_decay: L2 regularization to prevent overfitting (1e-3 to 1e-4 is common).
    # momentum: Helps accelerate gradients in the right direction (0.9 is industry standard).
    lr = 0.01
    weight_decay = 0.001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**1. Number of parameters:**

The number of parameters in a convolutional layer is given by:
$$C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w$$
where $C_{\text{in}}$ is the number of input channels, $C_{\text{out}}$ is the number of output channels, and $K_h, K_w$ are the kernel height and width.

Assume the regular block consists of two 3×3 convolutional layers, each with 256 input and 256 output channels.

Each 3×3 convolution has $256 \times 256 \times 3 \times 3 = 589,824$ parameters.

Total for regular block: $2 \times 589,824 = 1,179,648$ parameters.

The bottleneck block consists of:
- 1×1 convolution: 256 input → 64 output channels: $256 \times 64 \times 1 \times 1 = 16,384$ parameters.
- 3×3 convolution: 64 input → 64 output channels: $64 \times 64 \times 3 \times 3 = 36,864$ parameters.
- 1×1 convolution: 64 input → 256 output channels: $64 \times 256 \times 1 \times 1 = 16,384$ parameters.

Total for bottleneck block: $16,384 + 36,864 + 16,384 = 69,632$ parameters.

The bottleneck block has significantly fewer parameters due to the reduced channel dimension in the middle layer.

**2. Number of floating point operations:**

Convolutional layers' FLOPs are roughly proportional to $\text{number of conv layer parameters} \times H_{\text{out}} \times W_{\text{out}}$ (assuming the same output spatial dimensions for both blocks).

The regular block has two 3×3 convolutions, each with $256 \times 256 \times 9$ operations per spatial position, totaling approximately $2 \times 256 \times 256 \times 9 \times H_{\text{out}} \times W_{\text{out}} = 1,179,648 \times H_{\text{out}} \times W_{\text{out}}$ FLOPs.

The bottleneck block has:
- 1×1 convolution: $256 \times 64 \times 1$ operations per spatial position.
- 3×3 convolution: $64 \times 64 \times 9$ operations per spatial position.
- 1×1 convolution: $64 \times 256 \times 1$ operations per spatial position.

Totaling approximately $(256 \times 64 + 64 \times 64 \times 9 + 64 \times 256) \times H_{\text{out}} \times W_{\text{out}} = 69,632 \times H_{\text{out}} \times W_{\text{out}}$ FLOPs.

The bottleneck block requires significantly fewer FLOPs because the costly 3×3 convolution is performed on a reduced number of channels (64 instead of 256), and the 1×1 convolutions have smaller kernels, making them computationally inexpensive.

**3. Ability to combine the input:**

**(1) Spatially (within feature maps):** The regular block, with two stacked 3×3 convolutions, has a larger effective receptive field and performs more spatial mixing operations, allowing better combination of spatial information within feature maps compared to the bottleneck block's single 3×3 convolution on reduced channels.

**(2) Across feature maps:** The bottleneck block is better here due to the 1×1 convolutions, which efficiently combine information across all input channels. The regular block's 3×3 convolutions also combine across channels but are less efficient for this purpose compared to dedicated 1×1 layers.
"""


part4_q2 = r"""
**1.** Given $\frac{\partial L}{\partial y_1}$, derive $\frac{\partial L}{\partial x_1}$.

Since $y_1 = M x_1$, the Jacobian $\frac{\partial y_1}{\partial x_1} = M$.

Thus, $\frac{\partial L}{\partial x_1} = M^\top \frac{\partial L}{\partial y_1}$.

**2.** Given $\frac{\partial L}{\partial y_2}$, derive $\frac{\partial L}{\partial x_2}$.

Since $y_2 = x_2 + M x_2 = (I + M) x_2$, the Jacobian $\frac{\partial y_2}{\partial x_2} = I + M$.

Thus, $\frac{\partial L}{\partial x_2} = (I + M)^\top \frac{\partial L}{\partial y_2}$.

**3.** In deep networks with many layers, gradients are computed by multiplying Jacobians (or their transposes) along the chain rule.

For a non-residual layer ($y = M x$), the gradient flow is $\frac{\partial L}{\partial x} = M^\top \frac{\partial L}{\partial y}$. If $|M| < 1$ elementwise, repeated multiplication can cause gradients to vanish exponentially.

For residual layers ($y = x + M x$), the gradient is $(I + M)^\top \frac{\partial L}{\partial y}$. The identity term $I$ ensures that part of the gradient (at least 1) is preserved through each layer, preventing vanishing gradients even when $M$ has small entries.
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