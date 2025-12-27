import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator

def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.
    """
    # ====== YOUR CODE: ======
    # 1. Get unique characters using set() and sort them for lexical order
    unique_chars = sorted(list(set(text)))
    
    # 2. Create the mapping from char to index using a dictionary comprehension
    # enumerate gives us both the index and the character
    char_to_idx = {char: i for i, char in enumerate(unique_chars)}
    
    # 3. Create the mapping from index to char
    idx_to_char = {i: char for i, char in enumerate(unique_chars)}
    # ========================
    
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # ====== YOUR CODE: ======
    # Create a set of characters to remove
    to_remove = set(chars_to_remove)
    
    # Use a list comprehension to keep only allowed characters
    chars_list = [char for char in text if char not in to_remove]
    
    # Reconstruct the cleaned text
    text_clean = "".join(chars_list)
    
    # Calculate how many characters were filtered out
    n_removed = len(text) - len(text_clean)
    # ========================
    
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # ====== YOUR CODE: ======
    S = len(text)
    V = len(char_to_idx)
    result = torch.zeros((S, V), dtype=torch.int8)
    for i, char in enumerate(text):
        idx = char_to_idx[char]
        result[i, idx] = 1
    return result
    # ========================


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # ====== YOUR CODE: ======
    # Handle the batch dimension if present (B, S, V)
    if embedded_text.dim() == 3:
        # Get the indices of the '1's along the V dimension
        indices = torch.argmax(embedded_text, dim=-1)
        return ["".join([idx_to_char[idx.item()] for idx in seq]) for seq in indices]
    
    # Handle single sample (S, V)
    indices = torch.argmax(embedded_text, dim=-1)
    return "".join([idx_to_char[idx.item()] for idx in indices])
    # ========================


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # 1. Calculate how many full sequences we can create
    # Total chars available for samples is len(text) - 1 because the last char has no label.
    num_samples = (len(text) - 1) // seq_len
    total_len = num_samples * seq_len
    
    # 2. Convert text to indices first
    indices = torch.tensor([char_to_idx[c] for c in text], device=device)
    
    # 3. Create samples: Embed text characters up to the last full sequence
    # Note: We use the helper you'll implement or already have: chars_to_onehot
    # But to do it without loops as requested, we embed first then reshape.
    V = len(char_to_idx)
    
    # Input chars for samples: indices 0 to total_len-1
    sample_indices = indices[:total_len]
    # One-hot encoding the sample indices
    samples = torch.zeros((total_len, V), dtype=torch.int8, device=device)
    samples.scatter_(1, sample_indices.unsqueeze(1), 1)
    # Reshape to (N, S, V)
    samples = samples.view(num_samples, seq_len, V)
    
    # 4. Create labels: Shifted by one (indices 1 to total_len)
    label_indices = indices[1 : total_len + 1]
    # Reshape to (N, S)
    labels = label_indices.view(num_samples, seq_len)
    # ========================
    
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    # 1. Scale the scores by the temperature.
    # Note: Mathematically, dividing the vector by T before softmax 
    # implements the equation provided in the instructions.
    scaled_y = y / temperature
    
    # 2. Compute softmax along the specified dimension.
    # Using torch.softmax ensures numerical stability.
    result = torch.softmax(scaled_y, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().

    # Disable gradient tracking for inference to save memory and speed up computation
    with torch.no_grad():
        # ====== YOUR CODE: ======
        # 1. Initialize hidden state and process the starting sequence
        # We need to one-hot encode the start_sequence to feed it to the model
        curr_indices = torch.tensor([char_to_idx[c] for c in start_sequence], device=device)
        
        # Prepare one-hot input (S, V)
        V = len(char_to_idx)
        x = torch.zeros((len(start_sequence), V), device=device)
        x.scatter_(1, curr_indices.unsqueeze(1), 1.0)
        
        # Add batch dimension: (B=1, S, V)
        x = x.unsqueeze(0)
        
        # Feed the entire start sequence at once to get the initial hidden state
        # y: (1, S, V), h: (1, L, H)
        y, h = model(x)

        # 2. Start generating characters one by one
        for _ in range(n_chars - len(start_sequence)):
            # Use the last score output from the model to predict the NEXT char
            # last_scores shape: (V,)
            last_scores = y[0, -1, :]
            
            # Convert scores to probabilities using the temperature-scaled softmax
            # Use the hot_softmax function implemented earlier
            probs = hot_softmax(last_scores, dim=0, temperature=T)
            
            # Sample a character index based on the probability distribution
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Convert index back to character and append to output
            new_char = idx_to_char[sampled_idx]
            out_text += new_char
            
            # 3. Prepare the new char as input for the next step
            # Note: We only feed the single new char and the previous hidden state
            x_next = torch.zeros((1, 1, V), device=device)
            x_next[0, 0, sampled_idx] = 1.0
            
            # Forward pass with the single new char and the accumulated hidden state
            y, h = model(x_next, h)
        # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        n_samples = len(self.dataset)
        
        # Calculate how many batches we can fully complete
        num_batches = n_samples // self.batch_size
        
        # Total number of indices we will use (discarding the remainder)
        total_indices = num_batches * self.batch_size
        
        # Create the indices and arrange them so that indices are 
        # contiguous across batches for the same batch index.
        # We effectively transpose the index layout.
        indices = list(range(total_indices))
        
        # Reshape to (batch_size, num_batches) and then transpose to (num_batches, batch_size)
        # This way, index 0 of batch 0 is indices[0], index 0 of batch 1 is indices[1], etc.
        # But index 1 of batch 0 is indices[num_batches].
        idx = []
        for b in range(num_batches):
            for i in range(self.batch_size):
                # This formula ensures sample k in batch j is followed by 
                # sample k in batch j+1 (which is index + 1 in the original dataset)
                idx.append(i * num_batches + b)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        
        # Use nn.ModuleList so parameters are registered and moved to GPU
        self.layer_params = nn.ModuleList()

        # 1. Create the L actual GRU layers
        for i in range(self.n_layers):
            curr_in_dim = in_dim if i == 0 else h_dim
            
            # Each layer has 7 modules to match your forward unpacking
            layer_modules = nn.ModuleList([
                nn.Linear(curr_in_dim, h_dim, bias=True),  # update_wx
                nn.Linear(h_dim, h_dim, bias=False),       # update_wh
                nn.Linear(curr_in_dim, h_dim, bias=True),  # reset_wx
                nn.Linear(h_dim, h_dim, bias=False),       # reset_wh
                nn.Linear(curr_in_dim, h_dim, bias=True),  # candidate_wx
                nn.Linear(h_dim, h_dim, bias=False),       # candidate_wh
                nn.Dropout(dropout)                        # dropout_layer
            ])
            self.layer_params.append(layer_modules)
            
        # 2. Add an (L+1)-th entry for the final projection
        # This ensures self.layer_params[-1][0] is the correct linear layer
        # Your loop only goes up to n_layers, so it will never 'unpack' this entry
        output_entry = nn.ModuleList([
            nn.Linear(h_dim, out_dim, bias=True), # This is the [0] element used for output
            # Fill the rest with dummies so it doesn't break if anything else probes it
            nn.Identity(), nn.Identity(), nn.Identity(), 
            nn.Identity(), nn.Identity(), nn.Identity()
        ])
        self.layer_params.append(output_entry)

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # Loop over layers of the model

        activation_sigmoid, activation_tanh = nn.Sigmoid(), nn.Tanh()
        output_seq = torch.zeros_like(input)

        for time_step in range(seq_len):
            current_input = layer_input[:, time_step]

            for layer_index, prev_state in enumerate(layer_states):
                update_wx, update_wh, reset_wx, reset_wh, candidate_wx, candidate_wh, dropout_layer = self.layer_params[layer_index]

                if layer_index > 0:
                    current_input = layer_states[layer_index - 1]

                update_gate = activation_sigmoid(update_wx(current_input) + update_wh(prev_state))
                reset_gate = activation_sigmoid(reset_wx(current_input) + reset_wh(prev_state))
                candidate_state = activation_tanh(candidate_wx(current_input) + candidate_wh(reset_gate * prev_state))

                new_state = update_gate * prev_state + (1 - update_gate) * candidate_state
                layer_states[layer_index] = new_state

                if layer_index > 0:
                    current_input = dropout_layer(new_state)

            output_seq[:, time_step] = self.layer_params[-1][0](layer_states[-1])

        final_hidden_state = torch.stack(layer_states, dim=1)
        hidden_state = final_hidden_state
        layer_output = output_seq

        return layer_output, hidden_state
