r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=64,          # Larger batches provide more stable gradient estimates
        seq_len=64,             # Long enough to capture words and basic sentence structure
        h_dim=512,              # Increased hidden dimension for higher model capacity
        n_layers=3,             # Depth to learn hierarchical text representations
        dropout=0.2,            # Moderate dropout to prevent overfitting on the corpus
        learn_rate=0.001,       # Standard starting rate for the Adam optimizer
        lr_sched_factor=0.5,    # Reduce LR by half when performance plateaus
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
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
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
