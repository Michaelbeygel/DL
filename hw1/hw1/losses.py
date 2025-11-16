import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C). - scores = X @ W
        :param y_predicted: The predicted class label for each sample: (N,). - max in each row of x_scores
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: \\ Done
        # Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).
        N, C = x_scores.shape

        # Indices of samples: [0, 1, ..., N−1]
        sample_idx = torch.arange(N)

        # Extract the score of the correct class for each sample.
        # Shape: (N,) → reshape to (N,1) so broadcasting works against (N,C)
        correct_scores = x_scores[sample_idx, y].view(-1, 1)

        # Compute margin matrix:
        # M[i,j] = Δ + score(x_i, class j) − score(x_i, true class)
        M = self.delta + x_scores - correct_scores

        # Zero out margins for the true class j = y[i],
        # because we do not include them in the hinge sum.
        M[sample_idx, y] = 0.0

        # Apply hinge: max(M, 0)
        M = torch.clamp(M, min=0.0)

        # Compute per-sample hinge loss by summing over classes j ≠ y[i]
        sample_loss = M.sum(dim=1)   # shape: (N,)

        # Average over all samples → scalar (0-dim tensor)
        loss = sample_loss.mean()
        
        loss = loss.view(1) 
        # TODO: \\ Done
        # Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================
        self.grad_ctx = {
            "x": x,           # (N, D)
            "y": y,           # (N,)
            "margin": M,      # (N, C)
        }

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO: \\ Done
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.
        x = self.grad_ctx["x"]        # (N, D)
        y = self.grad_ctx["y"]        # (N,)
        M = self.grad_ctx["margin"]   # (N, C)

        N = x.shape[0]
        dtype = x.dtype

        # dL/dS = G, where G[i,j] = 1 if margin_ij > 0, else 0, and
        # G[i, y_i] = -sum_{j != y_i} 1{margin_ij > 0}
        G = (M > 0).to(dtype)                       # (N, C) in {0,1}
        row_pos_counts = G.sum(dim=1)               # (N,)
        G[torch.arange(N), y] = -row_pos_counts

        # dL/dW = X^T * G / N  because S = XW and we average over N
        grad = x.t().matmul(G) / N                  # (D, C)

        return grad
