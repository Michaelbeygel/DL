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

$$(N \times D_{\text{out}}) \times (N \times D_{\text{in}}) = (32,768) \times (65,536)$$

**1.2 Block Matrix Structure:**

When viewed as a block matrix with blocks of shape $(D_{\text{out}} \times D_{\text{in}}) = (512 \times 1024)$:

The Jacobian is a **diagonal block matrix**. The structure is:
- **Diagonal blocks** (where $i = j$): Each diagonal block equals $\mat{W}^T$ because sample $i$'s output depends only on sample $i$'s input $\text{s.t. } y_i = x_i W^T$
- **Off-diagonal blocks** (where $i \neq j$): All zeros, because the output of sample $i$ is independent of the input of sample $j$

**1.3 Optimization:**

Because the Jacobian is a diagonal block matrix whose diagonal blocks are all identical (each block equals $\mat{W}^\top$), we do not need to store every block separately. Instead, we can store a single copy of the weight matrix.

- **Optimized storage:** one copy of the weight matrix $\mat{W}$ (or $\mat{W}^\top$) with shape $(512) \times (1024)$.

This reduces storage from the full Jacobian of shape $(64 \times 512 \times 64 \times 1024)$  down to $(512\times1024)$, which is $64^2$ times smaller.


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

Explanation: For a single sample $i$ and output unit $p$ we have
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

Yes, it is possible. 
We will provide an example where both the loss and the accuracy of the model decrease.

First, let's take a look at the cross entropy loss for a given batch of size N:

$
\mathcal{L}
= -\frac{1}{N} \sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)
\right]
$

The cross entropy loss measures how much probability the model assigns to the true label.
It is sensitive to the confidence of the prediction.
Accuracy, on the other hand, depends only whether the predicted class matches the true lables.

With that intuition in mind, let's look at an example:
We will look at a binary classification problem.
- Epoch A:
    9 correct predictions, with high loss. 
    1 incorrect prediction.

    The correct predictions are correct because they passed some threshold, but still have low confidence in the true labels.
- Epoch B:
    8 correct predictions, with low loss.
    2 incorrect predictions.

    The model assign high probability to the correct predictions. 

The loss decrease from A to B because there are more predictions that are close to their true labels.
And the accuracy decrease as well.

Therefore it is possible for both the loss and the accuraacy to decrease.


    
"""

part2_q3 = r"""
**Your answer:**

1. Let's compare the differences and similarities of GD and SGD.

    Similarities:
    - They operate in a similar way(itertive updates). 
    - If they converge, it is for a local minimum in non-convex functions, and for a global minimum in convex functions.
    - Both of them use the gradient of the function to find a minimum.

    Differences:
    - GD uses the entire dataset to calculate the gradient for one iteration. 
    SGD uses a mini-batch(by definition it uses 1 sample, but in practice it is typically a mini batch...) instead.
    - Run time in SGD is faster per iteration.
    - SGD requires more iterations to converge.
    - SGD is more noisy, because it is influanced only by the small mini-batch data.
    - The randomness in SGD can help it get out of local minimum. Therefore GD is more likely to get stuck at a local minimum.

2. Momentum in regular GD might be helpful.

    If there is a big ratio gap between the axes of the data(the loss function surface looks like a narrow valley), then GD will suffer from "zig zag" behavior.
    Adding momentum will help us converge quickly to the minimum, and bypass the "zig zag".

    In addition to that, adding momentum might help us get out of small local minima.
    That is a simillar to the behaviour we saw in SGD with momentum. 

3. 
    1) In the original GD algorithm, each iteration updates its parameters in the following way:

    $$w_{(t+1)} = w_{(t)} - \eta \nabla_w L \quad , \text{ where } \quad \nabla_w L = \nabla_w \left( \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, \hat{y}_i) \right) = \frac{1}{N} \sum_{i=1}^{N} \nabla_w \ell(y_i, \hat{y}_i)$$

    Suppose we have a disjoint split of the data $\{N_i\}_{i=1}^{m} \text{ , } N = \left| \bigcup_{i=1}^{m} N_i \right|$.

    In the suggested GD algorithm, each forward pass would calculate:
    - $\sum_{y \in N_i} \ell(y, \hat{y})$ , the sum of the loss functions for the data in batch $N_i$.
    - Each layer saves the input data it received, for the backward pass.
    
    Because $\sum_{i=1}^{N} \ell(y_i, \hat{y}_i) = \sum_{i=1}^{m} \sum_{y \in N_i} \ell(y, \hat{y})$, from the linearity of the gradient we would get that:

    $\nabla_w \sum_{i=1}^{N} \ell(y_i, \hat{y}_i) = \nabla_w \sum_{i=1}^{m} \sum_{y \in N_i} \ell(y, \hat{y})$.
    
    And therefore the gradients will be the same.

2) The problem is that in the forward pass each layer saves the intermidiate activations in the RAM. 
    Therefore, altough we split the data we are implicitly saving it in the RAM.

3) We can solve this issue by automatically calculating the gradient of the batch after the forward pass, and save only that.
    The gradients are zeroed before processing the first batch, and parameters are updated only after all batches have been processed.
    Therefore for each batch we would have a single vector of the gradient and that's it. 
    From the answer to part 1 of the current question we could deduce that it would be the same as regular GD.

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

1. Let's explain what each type of error means.
    1. Optimization error:
    
        This error is caused by a bad optimizer(might be because bad hyper parameters).
    It is the gap between the best possible model in the hypothesis class and the model found by training.
        
    2. Generalization error:

        This error is caused by the model overfitting the training data.
    It is defined as the gap between the models performance on the train and test sets.
    We can look at it as a measure of how accuratly the model is able to predict outcomes for data it has never seen before.

    3. Approximation error:
    
        This error caused by the limitations of the model to fit the data.
    It is the gap between the best model in the hypothesis class and the true target function.
    Good accuracy often leades to small approximation loss.
    If our model is underfitting, then it has high approximation error.

2. Let's take a look at each of the errors from the previous section of this question.

    1. Our model does not have high optimization error.
    
        The given "moon" data the model is trained on is noisy by itself.
        Therefore the ~95% accuracy we get in the train set can assure us that the optimizer is fine.

    2. Our model have a little generalization error, but nothng major.

        We can see from the plots that around epoch 8 begins a gap between the test and train in both plots.
        This tells us that the model is slightly overfitting.
        But, because we are still getting good performances in the test sets, the generalization error is not high.

    3. Our model does not have high approximation error.

        The model gets good results on the data, and is able to fit it quite good.
        Therefore we can deduce that the hypothesis class is expressive enough to model the underlying structure of the data.

"""

part3_q2 = r"""
**Your answer:**

Example 1 - prefer to optimize FPR.

We would want to minimize False Positive Rate, if false positives have bad consequences.

For example, suppose our model predicts whether or not a patient is free to leave the hospital.
A false positve result would mean that we let an ill person leave the hospital. 
He is then not treated as he should be, and might spread his disease!
On the other hand, we can compromise on high False Negative Rate, which means that we keep a patient that can already leave the hospital.
This will not have fatal consequences and therfore we care less about that.

Example 2 - prefer optimize FNR.

We would want to minimize False Negative Rate, if false negatives have bad consequences.

For example, suppose our model predicts whether or not a patient need to take some medicine.
A false negative would mean that a patient will not take his medicine.
A false positive would mean that a patient will take a medicine he do not need to take.
We assume that the medicine cannot cause health issues, but not taking it is unhealthy.
Therefore we would prefer to optimize the FNR at the cost of increasing FPR.

"""

part3_q3 = r"""
**Your answer:**

1. Each column represents a fixed depth.

    As we can see from the plots, in every column there is a gradual increase in model complexity as the model gain more width.
As a result, the decision boundaries evolve from simply linear to more complex areas.
This corresponeds to the Universal Approximation Theorem, which states that there exists a network with a single finite layer that can fit every continuous function "arbitrarily well".
We see this clearly in the first column(depth=1).
The performance also increase as the width is getting larger, until a certain point where its starts to decrease as a result of overgitting or optimization problems.

2. Each row represents a fixed width.

    As we can see from the plots, there is not a lot of change in decition boundaries between the different depth in each row.
In particular, we can notice that the best test accuracy is achieved by:
    - Row 1 (Width=2): Column 3 (Depth=4)
    - Row 2 (Width=8): Column 2 (Depth=2)
    - Row 3 (Width=32): Column 1 (Depth=1)

    This suggest that as the width increase, adding depth does not necessarily improve performance, and might actually hurt it.
For the 2D "moon" data, a single wide layer is sufficient, and gets the best test accuracy.
Also, we might get that the optimizer is having problems optimizing the deeper networks(duo to vanishing gradients for example).

3. We got very similar performances between the two models.

    Adding depth to the model, adds sequential non-linear activation functions.
As a result, altough both of the models have the same number of parameters, the deeper one might represent more complex datasets(note that the hypothesis classes are different). 
Because the data is not very complex, a single wide layer is sufficient.
Therefore both models get very similar accuracies, and the shallow one has slightly better generalization.

4. Threshold selection improved the results on the test set.

    As we saw earlier in this part, it help us get a better balance between FPR and FNR, and better performances in general.
That is duo to better interpretation of the outputs.
In the model experiment, each model chose a difference threshold, which improved test set resaults.
Some models chose a threshold as low as 0.12, which is quite far then the standard thresh=0.5.


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
### 1. Analysis of the Effect of Depth ($L$)
In this experiment, we observed that increasing the depth of a **plain** CNN (without skip connections) does not linearly improve performance.

* **Optimal Depth:** Typically, the best results are achieved with value of $L=4$. At these depth, the network has enough capacity to learn meaningful hierarchical features from the CIFAR-10 dataset while remaining shallow enough for the gradient to flow effectively from the output back to the input layers.
* **Performance Degradation:** As $L$ increases to 8 and 16, the network's accuracy begins to drop. Even though a deeper model has more parameters and theoretically higher representational capacity, the difficulty of optimizing a deep "plain" stack outweighs the benefits of the added layers.

### 2. Non-trainable values of $L$ and the Vanishing Gradient
We observed that for higher values of $L$ (specifically $L=8$ and $L=16$), the network often became **untrainable**. This was characterized by flat-line accuracy (near 10%) and loss curves that did not decrease, eventually triggering **early stopping**.

**The Cause: Vanishing Gradients**
In a plain CNN, the gradient is computed using the chain rule. During backpropagation, the error signal is multiplied by the weights and the derivatives of the activation functions at every layer. 
If intermediate derivatives are small (which is common with standard initialization), multiplying them 8 or 16 times causes the gradient to decay exponentially until it vanishes. The early layers receive no update, so the model never learns to extract basic features.

**Suggested Resolutions:**
1.  **Residual Connections (Skip Connections):** Introduce "shortcuts" that allow the gradient to bypass layers. By changing the layer to $y = F(x) + x$, the gradient can flow through the identity path ($+1$ in the derivative), ensuring it stays strong even at $L=16$.
    
2.  **Batch Normalization:** Adding Batchnorm layers after convolutions helps keep the activations in a range where the derivatives (e.g., of a ReLU or Tanh) are less likely to vanish or saturate, stabilizing the distribution of inputs to deeper layers and allowing for higher learning rates.
"""

part5_q2 = r"""

### Analysis of Experiment 1.2: Filter Width ($K$) vs. Depth ($L$)

#### 1. Case $L=2$:
* The network is shallow enough to converge across all tested filter widths $K$, but we observe that **smaller filter counts ($K=32$) generalize better in this configuration.
* **U-Shape Loss:** The test loss exhibits a distinct **U-shape**, reaching a minimum before rising. This indicates that as training continues, the added complexity of 64 or 128 filters introduces more parameters than the architecture can effectively regularize, causing the model to move past the point of optimal generalization.
* **Overfitting:** Significant overfitting is evident as training accuracy approaches $90\%$ while test accuracy plateaus much lower, creating a visible generalization gap.

#### 2. Case $L=4$:
At $L=4$, the relationship between width and performance flips; the network now requires more filters to reach its peak potential, though overfitting becomes even more pronounced.
* $K=64$ and $K=128$ outperformed $K=32$ in test accuracy.
* With 4 layers, the model has enough architectural depth to benefit from a higher number of features ($K=64, 128$). However, without regularization, the model uses its increased capacity to memorize the training set rather than generalizing.
* **U-Shape Loss:** A very sharp **U-shape** is visible in the test loss for all $K$ values. The loss drops quickly but rebounds aggressively after approximately iteration 7. This confirms that higher capacity leads to faster divergence once the training set is memorized.
* **Overfitting:** We observe extreme overfitting across all values of $K$. For $K=128$, training accuracy reaches nearly $100\%$, yet test accuracy remains around $70\%$, resulting in a massive generalization gap.

#### 3. Case $L=8$:
* All configurations ($K=32, 64, 128$) remained stuck at $\approx 10\%$ accuracy.
* This confirms that **width is irrelevant when depth creates an optimization barrier.** In a plain CNN architecture without skip connections or normalization, the vanishing gradient problem prevents any learning from occurring at $L=8$, rendering the number of filters $K$ moot.

### Comparison to Experiment 1.1
Comparing these results to Experiment 1.1 highlights how width and depth interact:
1. **Depth is the primary constraint:** Just as in Experiment 1.1, increasing depth beyond a certain point ($L=8$) leads to total failure that no amount of width ($K$) can fix.
2. Experiment 1.2 shows that the "best" $K$ depends on depth—$K=32$ was best for $L=2$, but $K=64$ was required to maximize the potential of $L=4$.
3. While Experiment 1.1 focused on the depth-limit, Experiment 1.2 shows that even at stable depths, adding width ($K=128$) or depth ($L=4$) accelerates overfitting, leading to high training accuracy but stagnant or degrading test performance due to the lack of regularization.
"""

part5_q3 = r"""

### Analysis of Experiment 1.3: Multi-Stage Filter Width ($K=[64, 128]$) vs. Depth ($L$)

#### 1. Case $L=2$ and $L=3$:
Both the $L=2$ and $L=3$ configurations successfully converged, as the network depth was shallow enough for the gradient to propagate effectively despite the increased width.
* Both models achieved high performance, with $L=2$ performing slightly better, peaking at approximately $73-74\%$ test accuracy, while $L=3$ reached approximately $69-70\%$.
* Both configurations exhibit significant overfitting, characterized by a large gap between training accuracy (approaching $100\%$) and test accuracy.
* A clear **U-shape** is visible in the test loss for both runs; the loss reaches a minimum before rebounding upward.
* This confirms that as the model exhausts its ability to generalize, it uses its high filter capacity to memorize the training set, causing the test performance to degrade. While $L=3$ provides more parameters, the optimization difficulty of the extra layer results in slightly lower accuracy than the $L=2$ baseline.

#### 2. Case $L=4$:
At $L=4$, the multi-stage architecture suffers a total training failure, highlighting the critical trade-off between filter width and network depth.
* Both training and test accuracy flatline at exactly $10\%$, and the training loss remains stagnant near $2.30$.
* This result demonstrates that **increasing the width incrementally ($64 \to 128$) cannot compensate for excessive depth** in a plain CNN architecture.
* Even though $L=4$ was trainable with narrower filters in previous runs, the added complexity of wider filters at this depth triggers an earlier optimization collapse. The gradients fail to propagate through the (K*L)=8-layer wide stack.

"""

part5_q4 = r"""

### Analysis of Experiment 1.4: The Impact of Skip Connections

In this experiment, we introduced skip connections (Residual connections) to resolve the optimization bottlenecks identified in previous experiments. These results clearly demonstrate that skip connections are the primary driver for training deeper networks effectively.

The most significant result is the successful training of deep architectures that previously suffered from total optimization collapse in Experiments 1.1 and 1.3.
* **Overcoming Vanishing Gradients:** Unlike plain CNNs, where $L \ge 8$ resulted in stagnant $10\%$ accuracy, the Residual networks with $L=8, 16, \text{and } 32$ converged successfully.
* **Identity Mapping:** Skip connections allow the gradient to bypass weight layers through the identity path $H(x) = F(x) + x$. This ensures that the error signal remains strong enough to update early layers even at extreme depths like $L=32$.

#### 1. Case $K=32$ ($L=8, 16, 32$):
* **The Benefit of Depth:** With skip connections, we see that $L=16$ (blue) achieves high test accuracy ($\approx 77\%$), outperforming the best results from Experiment 1.1.
* **Diminishing Returns:** Interestingly, the $L=32$ model (orange) performs worse than $L=16$, plateauing around $68\%$ test accuracy. This suggests that while skip connections solve trainability, very deep plain-residual stacks may still struggle with generalization or require additional stabilization like Batch Normalization to leverage their full capacity.

#### 2. Case $K=[64, 128, 256]$ ($L=2, 4, 8$):
This wide, multi-stage architecture achieved the best performance across all experiments.
* The $L=4$ variant reached a peak test accuracy of $\approx 80\%$, proving that combining skip connections with increased width ($K$) provides the most robust architecture for CIFAR-10.
* For $L=2$ and $L=4$, we observe a very sharp **U-shape** in the test loss. Because these models reach near $100\%$ training accuracy quickly, they begin to memorize the dataset, causing test loss to rebound aggressively after the initial drop.

### Comparison to Previous Experiments

1. **Comparison to Experiment 1.1:** In Experiment 1.1, depth was a liability beyond $L=4$. In Experiment 1.4, depth becomes an asset up to $L=16$, thanks to the stabilization provided by skip connections.
2. **Comparison to Experiment 1.3:** Experiment 1.3 showed that increasing width made deep networks even more fragile, with failure occurring at $L=4$. In contrast, the Residual version of the wide architecture ($L=8, K=[64, 128, 256]$) trains perfectly and outperforms all plain models.
3. **Generalization Gap:** Across all successful runs, the generalization gap remains the biggest challenge. While skip connections solved the **trainability wall**, the models still require early stopping or better regularization to manage the memorization of training noise.
"""

part5_q5 = r"""
### Question 5: Architecture Analysis of YourCNN and Experiment 2

#### 1. Architectural Enhancements in YourCNN
To improve training stability and performance relative to the initial "plain" CNN models, we introduced several key architectural changes in the `YourCNN` class:

* **Residual (Skip) Connections:** We utilized the `ResidualBlock` to implement skip connections ($H(x) = F(x) + x$). This allows gradients to bypass the weight layers through an identity path, fundamentally solving the vanishing gradient problem and allowing us to train deeper stacks that previously failed.
* **Integrated Batch Normalization:** We enabled `batchnorm=True` within each block to normalize feature map statistics. This stabilizes the distribution of inputs to deeper layers, prevents activation drifting, and allows for more robust optimization.
* **Strategic Dropout (0.2):** We applied a dropout rate of $0.2$ to provide regularization without starving the deep layers of signal. While higher dropout (0.5) was initially tested, the $0.2$ rate provided the necessary balance to prevent neuron co-adaptation while maintaining information flow in deep architectures.
* **Leaky ReLU Activation:** We replaced standard ReLU with Leaky ReLU (slope=$0.01$). This ensures a small gradient flow for negative inputs, mitigating the "dying ReLU" problem and preserving signal during the backward pass.



#### 2. Analysis of Experiment 2 Results
We evaluated this architecture using fixed pyramidal filters ($K=[32, 64, 128]$) across varying depths $L=3, 6, 9, 12$.

* **Successful Convergence at Extreme Depths:** The introduction of residual connections and BatchNorm allowed all configurations to train effectively. Unlike previous experiments where $L=8$ or $L=12$ failed, these models show consistently decreasing training loss and increasing accuracy.
* **Performance Peak at $L=3$:** The shallowest configuration ($L=3$, orange) achieved the highest overall performance, peaking at over $80\%$ test accuracy. As depth increased, we observed a steady decline in test performance, with $L=12$ (blue) reaching approximately $68\%$.
* **Generalization and Overfitting:** All models reached high training accuracy (between $72\%$ and $90\%$), but the gap between training and test accuracy widened with depth. The test loss converge after 17-20 epochs from that point test accuracy the same, but the traning accuracy improved which indicate start of overfitting.



#### 3. Comparison to Experiment 1.4 (Residual CNNs)
The `YourCNN` architecture is very similar to the Residual CNNs tested in Experiment 1.4, as both rely on skip connections to enable deep learning. However, this experiment places a stronger emphasis on **generalization** through the following improvements:

* **Improved Baseline:** The $L=3$ configuration in `YourCNN` reached $\approx 81\%$ test accuracy, outperforming the best $K=32$ results from Experiment 1.4 ($\approx 78\%$). This improvement is largely attributed to the addition of **Dropout**, **Batch Normalization** and **Leaky ReLU**, which provide better generalization and internal stability than the basic residual structure alone.
* **Balanced Regularization:** By making the dropout rate to $0.2$, `YourCNN` maintains a more consistent signal than the models in Experiment 1.4.
* Experiment 1.4 proved that skip connections fix **trainability**, but `YourCNN` demonstrates that combining them with Dropout, BatchNorm and Leaky ReLU is necessary to improve **generalization** on the test set. 
"""
# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1. The model detection of the objects is bad.

    In both images, the bounding boxes the model produces where not very good. 
In the first image, he interpreted only a part of the dolphin as an object - not all of the dolphin.
In the second image, he didnt produce a bounding box for the real cat.

    The model failed to classify correctly on almost all of the bounding boxes(failed on +80% of the boxes).
Moreover, he got high confidence in some of these false classification.
For example, in the first picture he classify the dolphins as humans with confidence of 0.9.
Which make it hard to rely on such a model.

2. There are several possible reasons for the model failures, one of them is:

    Domain shift. 
The model might have trained on clean images of dolphins, and not images on dolphins that are facing the sun. 
Or, it might be even that the training data did not had a dolphin class at all.
Also, it might have trained on dogs that have more "dogs characteristics" then the provided Shiba Inu dogs(which are a little similar to cats).
Therefore, because both of the photos are a little unique, it might be that the problems in classifications is duo to a domain shift.

    To resolve this, we should perform fine-tuning.
We can train the model with a richer dataset that would include pictures of doplhins facing the sun, Shiba Inu dogs, and more unique pictures.

    Another possible way to resolve this issue is data augmantation.
For example, in the dolphins picture, we might be able to increase lighting such that we could see the skin color of the dolphins.
Then, the model might identify the dolphins as dolphins and not as humans.

3. A PGD attack on an Object Detection model would itertively insert noise to input images in order to minimize performances.
This includes minimzing the right class classification confidance, and bounding box selection.
This will lead to missed detections or confidence misclassifications.

"""


part6_q2 = r"""
**Your answer:**


"""


part6_q3 = r"""
**Your answer:**

The model failed to detect correctly in all of the 3 attached pictures.
We will explain the main Object Detection pitfalls the model experienced with each picture.

Picture 1: Occlusion.
As we can see, the cat is sitting behind many plants that partially blocks it.
The model failed therefore to classify the cat correcly, and wrongly detects an apple in the image.

Picture 2: Model Bias.
The cat is not in its casual setting. It is underwater, and the light is bluish.
The model was likely trained on images of cats that are not near water, so this represent a domain shift.
As a resualt, in the given settings, the model is having problems classifying the cat correctly.
Note that a lot of bears do live near water resources. 
Therefore it is possible that the trained data contained many pictures of bears with similar characteristics to this pictures(blue tint, watery textures and more..).
This may explain why the model associated the features with the bear class.

Picture 3: Illumination conditions.
The cat picture now is overexposed.
It created two problems for the model:
1. Texture loss, as the picture is a little blurred and many of the pixels turned into pure white.
2. The model is probably not trained on overexposed picture. 
Therefore this picture is out of the original data distribution.

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