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


<font color='red'>**- Boaz notes:**</font>
(The question is about the shape, not the size.) 

By simple calculations we would get that:

The size of $\mat{X}$ is $64 \times 1024$.

The size of $\mat{Y}$ is $64 \times 512$.

The jacobian tensor would have to take into account the partial deriviative of each two elements from $\mat{X}$ and $\mat{Y}$.

Therefore the shape of the jacobian tensor will be $(64 \times 512 \times 64 \times 1024)$.

**1.2 Block Matrix Structure:**

When viewed as a block matrix with blocks of shape $(D_{\text{out}} \times D_{\text{in}}) = (512 \times 1024)$:

The Jacobian is a **diagonal block matrix**. The structure is:
- **Diagonal blocks** (where $i = j$): Each diagonal block equals $\mat{W}^T$ because sample $i$'s output depends only on sample $i$'s input <font color='red'> s.t : $Y_i = X_iW^T$ </font>
- **Off-diagonal blocks** (where $i \neq j$): All zeros, because the output of sample $i$ is independent of the input of sample $j$

**1.3 Optimization:**

Because the Jacobian is a diagonal block matrix whose diagonal blocks are all identical (each block equals $\mat{W}^\top$), we do not need to store every block separately. Instead, we can store a single copy of the weight matrix.

- **Optimized storage:** one copy of the weight matrix $\mat{W}$ (or $\mat{W}^\top$) with shape $(512) \times (1024)$.

This reduces storage from the full Jacobian size of $(32,768)\times(65,536)$ elements down to $512\times1024 = 524{,}288$ elements.

<font color='red'>**Boaz notes:**

This reduces storage from the full Jacobian of shape $(64 \times 512 \times 64 \times 1024)$  down to $(512\times1024)$, which is $64^2$ times smaller.

</font>

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

1. First, we can observe that the model without dropout do not overfit in earlier epochs.
Becuase the initial model do not overfit, adding dropout would probably mainly hurt capacity, rather then add generality.
Also, we would expect that the model would converge slower. 
That is because with dropout fewer neurons are active per batch, which makes it harder to fit the train data and slows loss minimization.

    The graphs of no-dropout vs dropout match the intuition explained above.

    From the accuracy plots, we see that training accuracy decreases as dropout increases, as expected.
Without dropout, the model starts to slightly overfit around epoch 15.
Around epoch 15, the model with dropout=0 stops improving test-set wise, and the model with dropout=0.4 keeps improving.
Therefore because of the slight overfit, a small dropout(0.4) does help us get better generality.

    Additionally, the train_loss graph shows that higher dropout leades to slower convergence, as expected.

    Lastly, note that when the dropout is too big, as in dropout=0.8, our model is very weak and not learns a lot, resulting in underfitting.

2. Let's compare the low-dropout setting to the high-dropout setting.

    Dropout is a part of regularization. 
    Regularization encourages better generalization. 
    But, adding "too much" might make it very hard for the model to learn and therefore cause underfitting. 
    As a result, when adding regularization we would want to be in the "sweet spot" between under and over fitting.
    
    We can see from the graphs, that the model with dropout=0.8 result in bad losses and accuracies, both in the train and test sets.
That because it adds too much regularization, as explained above. 
Also we can see that the loss function of that model does not decrease. 
That strengthens our intuition that it is very hard for the model to learn in that condition. 
    
    In comparison, the model with dropout=0.4 have higher accuracies in the train and test sets. 
Since the training and test accuracies are similar across epochs, we can deduce that the model do not overfit.
Also, the loss functions in the train and test sets seems to have a gradually improve over the epochs, which the regularization is not too strong.

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
    n_layers = 3 # number of layers (not including output)
    hidden_dims = 64   # number of output dimensions for each hidden layer
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
    lr, weight_decay, momentum = 0.001, 1e-4, 0.95
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

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
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