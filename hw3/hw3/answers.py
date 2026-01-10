r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=256,          # Larger batches provide more stable gradient estimates
        seq_len=64,             # Long enough to capture words and basic sentence structure
        h_dim=128,              # Increased hidden dimension for higher model capacity
        n_layers=3,             # Depth to learn hierarchical text representations
        dropout=0.25,            # Moderate dropout to prevent overfitting on the corpus
        learn_rate=0.001,       # Standard starting rate for the Adam optimizer
        lr_sched_factor=0.1,    # Reduce LR by half when performance plateaus
        lr_sched_patience=3,    # Wait 3 epochs of no improvement before reducing LR
    )
    # ====== YOUR CODE: ======
    # These values are tuned for a Shakespearean char-RNN task.
    # A larger h_dim (512) and 3 layers allow the model to learn 
    # the complex patterns of old-style English.
    # ========================
    return hypers


def part1_generation_params():
    # ====== YOUR CODE: ======
    # Providing a classic Shakespearean opening helps the model
    # stay within the desired context (Acts, Scenes, and Characters).
    start_seq = "ACT I. Scene I. Elsinore. A platform before the Castle."
    
    # A temperature between 0.5 and 0.8 is usually ideal.
    # 0.0001 is too low and will likely result in a repetitive loop of the same word.
    # 0.5 makes the model confident but still allows for diverse word choices.
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
Training on the entire corpus as a single sequence is practically impossible due to hardware and algorithmic constraints. 

First, **memory limitations** are a primary factor; the "Backpropagation Through Time" (BPTT) algorithm requires storing the activations (hidden states) for every single timestep in a sequence to calculate gradients. If we used the whole corpus, the computational graph would grow so large that it would exceed the available GPU/CPU memory. 

Second, splitting the text allows us to use **mini-batching**, which parallelizes the training process by performing matrix-matrix multiplications instead of slower matrix-vector operations. 

Finally, training on shorter sequences helps mitigate the **vanishing/exploding gradient problem** common in RNNs. By using truncated sequences, we "break" the gradient flow at reasonable intervals, ensuring the model can still learn effectively without the gradients becoming numerically unstable over millions of characters.
"""

part1_q2 = r"""
The model is able to exhibit memory longer than the sequence length `S` because the hidden state is preserved and passed between contiguous batches. 

During training, we use a `SequenceBatchSampler` which ensures that the $k$-th sample in batch $j$ is the direct continuation of the $k$-th sample in batch $j-1$. In our `RNNTrainer`, we store the `hidden_state` at the end of a forward pass and feed it as the initial state for the next batch. This process, known as Truncated Backpropagation Through Time (TBPTT), allows the hidden state to accumulate information and context from the very beginning of the text stream, even though gradients only flow back through the current `seq_len`.

Similarly, during generation, the `generate_from_model` function passes the updated hidden state `h` into the next forward pass for every single character produced. This allows the "memory" of the model—encoded in the hidden state—to carry over across thousands of characters, enabling it to maintain consistent character names and stage directions far beyond the original training sequence length.
"""

part1_q3 = r"""
We do not shuffle the batches because our training strategy relies on the contiguous nature of the sequences to maintain long-term context. 

The `SequenceBatchSampler` is specifically designed so that the $k$-th sample of batch $i$ is the direct successor of the $k$-th sample in batch $i-1$. If we were to shuffle the batches, this sequential relationship would be destroyed. Since our `RNNTrainer` carries the hidden state forward from one batch to the next to allow the model to learn dependencies longer than the `seq_len`, shuffling would cause the model to receive a hidden state representing one part of the text while processing a completely unrelated segment from a different part of the corpus. 

Essentially, shuffling would turn our stateful RNN into a stateless one, where the hidden state becomes "noise" rather than a meaningful memory of the preceding text, severely hindering the model's ability to learn Shakespearean structure and long-term coherence.
"""

part1_q4 = r"""
1. **Lowering the Temperature**: We lower the temperature (e.g., to $0.5$) to make the model more "confident" and less likely to sample low-probability characters. This sharpens the distribution, increasing the probability of characters with the highest scores and reducing the chance of sampling characters that might be grammatically or contextually incorrect, leading to more coherent and readable text.

2. **High Temperature**: When the temperature is very high ($T \to \infty$), the distribution becomes nearly uniform. This happens because dividing the scores by a very large number makes them nearly equal before the softmax is applied. As a result, every character in the vocabulary becomes almost equally likely to be sampled, regardless of its score, which typically results in gibberish or complete randomness.

3. **Low Temperature**: When the temperature is very low ($T \to 0$), the distribution becomes extremely "sharp" or "pointy," eventually approaching a one-hot distribution centered on the character with the highest score. This occurs because the differences between the scores are amplified significantly by the small divisor. While this makes the output very structured, it often causes the model to become repetitive or stuck in infinite loops, as it loses the "creativity" needed to transition between different types of sentence structures.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=32, h_dim=512, z_dim=128, x_sigma2=0.0005, learn_rate=0.0002, betas=(0.85, 0.998),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


part2_q1 = r"""

We assume that: $p _{\mathbb{\beta}}(\mathbb{X} | \mathbb{Z}=\mathbb{z}) = \mathcal{N}( \Psi _{\mathbb{\beta}}(\mathbb{z}) , \sigma^2 \mathbb{I} )$.

That is because given a latent point $z$, we map it to an instance point normally distributed around the decoder of z.

$\sigma^2$ is the variance of that normal distribution.

$\sigma^2$ appears in the VAE loss function in the part:

$$ 1) \quad \quad
\mathbb{E}_{\mathbb{x}}
\left[
\mathbb{E}_{\mathbb{z} \sim q_{\vec{\alpha}}(\mathbb{z}\mid\mathbb{x})}
\left[
-\log p_{\vec{\beta}}(\mathbb{x}\mid\mathbb{z})
\right]
\right]
$$

Or in an equivalent way(after applying the reparmetrization trick):

$$ 2) \quad \quad
\mathbb{E}_{\mathbb{x}}
\left[
\mathbb{E}_{\mathbb{z} \sim q_{\vec{\alpha}}}
\left[
\frac{1}{2\sigma^2} 
\left\| 
\mathbb{x} - \Psi_{\vec{\beta}}\Big( 
\mathbb{\mu}_{\vec{\alpha}}(\mathbb{x}) + \mathbb{\Sigma}_{\vec{\alpha}}^{\frac{1}{2}}(\mathbb{x}) \mathbb{u} 
\Big) 
\right\|_2^2
\right]
\right]
$$


In representation $1)$, $\sigma^2$ is embedded in $p_{\vec{\beta}}(\mathbb{x}\mid\mathbb{z})$ because 
$p _{\mathbb{\beta}}(\mathbb{X} | \mathbb{Z}=\mathbb{z}) = \mathcal{N}( \Psi _{\mathbb{\beta}}(\mathbb{z}) , \sigma^2 \mathbb{I} )$.
Therefore the higher $\sigma^2$ will be, the noisier $p_{\vec{\beta}}(\mathbb{x}\mid\mathbb{z})$ will get.

In representation $2)$, $\sigma^2$ is embedded in $\frac{1}{2\sigma^2}$.
Therefore higher $\sigma^2$ control how much to emphasize on the data-loss in contrast to the KL divergence.

Therefore:
- Low $\sigma^2$: the model is heavily penalized for any pixel difference between input and output.
As a result, the reconstructions will be very sharp and similar to the input.
- High $\sigma^2$ will enforce high emphasize on KL-Divergance. 
The outputs generated by the decoder will lose details, and will not be very similar to the input.
In addition the model will be more robust to noisy data and overfitting.
"""

part2_q2 = r"""

1. 
**Reconstruction loss**:

$$
\mathbb{E}_{\mathbb{z} \sim q_{\vec{\alpha}}}
\left[
\frac{1}{2\sigma^2} 
\left\| 
\mathbb{x} - \Psi_{\vec{\beta}}\Big( 
\mathbb{\mu}_{\vec{\alpha}}(\mathbb{x}) + \mathbb{\Sigma}_{\vec{\alpha}}^{\frac{1}{2}}(\mathbb{x}) \mathbb{u} 
\Big) 
\right\|_2^2
\right]
$$

Such that $z \sim q_\alpha$ by the reperametrization trick, 
$z=\mathbb{\mu}_{\vec{\alpha}}(\mathbb{x}) + \mathbb{\Sigma}_{\vec{\alpha}}^{\frac{1}{2}}(\mathbb{x}) \mathbb{u}$
where $\mathbb{u}\sim\mathcal{N}(\mathbb{0},\mathbb{I})$.

Therefore it is equivalent to:
$$
\mathbb{E}_{\mathbb{z} \sim q_{\vec{\alpha}}}
\left[
\frac{1}{2\sigma^2} 
\left\| 
\mathbb{x} - \Psi_{\vec{\beta}}(z) 
\right\|_2^2
\right]
$$
Minimizing it will lead to smaller $\left\| \mathbb{x} - \Psi_{\vec{\beta}}(z) \right\|_2^2$ 
and therefore to smaller difference between inputs and decoded latent space points.

This part is whishes to make the overall encoder to decoder mapping as close to the identity map as possible.

**KL divergence loss**:
$$\mathcal{D} _{\mathrm{KL}}\left(q _{\mathbb{\alpha}}(\mathbb{Z} | \mathbb{x})\,\left\|\, p(\mathbb{Z} )\right.\right)$$

With the reconstruction loss alone, our model might suffer from overfitting.

The porpuse of the KV therefore is to add regularization.
Specifically it enforces the encoder to learn a smooth $q(z\|x)$, which will be as similar as possible to $p(\mathbb{Z})= \mathcal{N}(\mathbb{0},\mathbb{I})$.


2. 
Mathematically, the KL loss is defined by:
$$\mathcal{D} _{\mathrm{KL}}\left(q _{\mathbb{\alpha}}(\mathbb{Z} | \mathbb{x})\,\left\|\, p(\mathbb{Z} )\right.\right) =
\mathbb{E}_{\mathbb{z} \sim q_{\vec{\alpha}}}\left[ \log \frac{q _{\mathbb{\alpha}}(\mathbb{Z} | \mathbb{x})}{p(\mathbb{Z})} \right] =
\mathbb{E}_{\mathbb{z} \sim q_{\vec{\alpha}}}\left[ \log q_{\mathbb{\alpha}}(\mathbb{Z} | \mathbb{x}) - \log p(\mathbb{Z}) \right]$$

Therefore, minimizing it will make the latent-space-posterior distribution as close to $p(\mathbb{Z}) = \mathcal{N}(\mathbb{0},\mathbb{I})$ as possible.

3.
The main benefit of this effect is that we can generate new data from a normal distribution using the decoder.

We are enforcing the latent posterior to look a certain way, this increase generalization but also gurentee us that the decoder will train on data that is similar to a normal distibution.
Therefore the decoder learns to map points from this approximatly normal distribution back to the original data distribution.


"""

part2_q3 = r"""
We want to train our model to generate data that follows the distribution of the given input instance data.
Maximizing the evidence distribution $p(X)$ means to maximize the probability to get each $x\in X$.
Therefore maximizing $p(X)$ will make our model learn the input instance data distribution. 
And that is exactly what we want.
"""

part2_q4 = r"""
We model the log-variance in order to enforce positivoty constraints and ensure numerical stability.
- Positivity constraints: Variance must hold that $\sigma^2 > 0$. 
By predicting $\log \sigma^2$ we would get that $\sigma^2 = \exp ( \log \sigma^2)$ which is always positive.
- Numerical stability: Predicting variance directly would require constrained outputs, which leads to unstable or biased gradients. 
Using log-variance allows unconstrained optimization and would lead to more stable gradients.
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 128, 
        num_heads = 4,
        num_layers = 3,
        hidden_dim = 256,
        window_size = 128,
        droupout = 0.2,
        lr=0.0001,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
