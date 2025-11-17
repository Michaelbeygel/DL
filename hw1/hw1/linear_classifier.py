import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: \\ Done
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = torch.normal(mean = 0, std=weight_std, size=(n_features, n_classes))

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: \\ Done
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        class_scores = x @ self.weights # (N, n_classes)

        # Take the highest-score class per row
        y_pred = torch.argmax(class_scores, dim=1)

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: \\ Done
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        # Compare element-wise → boolean tensor
        correct = (y == y_pred)

        # Compute mean → fraction correct
        acc = correct.float().mean()

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            average_loss = 0
            train_num_batches = 0
            for x, y in dl_train:
                y_pred, x_scores = self.predict(x)
                loss = loss_fn(x, y, x_scores, y_pred) + 0.5 * weight_decay * torch.sum(self.weights * self.weights)
                grad = loss_fn.grad() + weight_decay * self.weights
                self.weights -= learn_rate * grad

                # ---- Train stats ----
                average_loss += loss.item()                     # sum of batch mean losses
                train_num_batches += 1

            # ---- Epoch train metrics ----
            average_loss /= train_num_batches
            train_accuracy = self.evaluate_accuracy(y, y_pred).item()

            train_res.loss.append(average_loss)
            train_res.accuracy.append(train_accuracy)

            # ===== VALIDATION PHASE =====
            val_average_loss = 0
            val_num_batches = 0

            for x_val, y_val in dl_valid:
                y_pred_val, x_scores_val = self.predict(x_val)
                loss_val = loss_fn(x_val, y_val, x_scores_val, y_pred_val) + 0.5 * weight_decay * torch.sum(self.weights * self.weights)

                # ---- Val stats ----
                val_average_loss += loss_val.item()
                val_num_batches += 1

            # ---- Epoch validation metrics ----
            val_average_loss /= val_num_batches
            val_accuracy = self.evaluate_accuracy(y_val, y_pred_val).item()

            valid_res.loss.append(val_average_loss)
            valid_res.accuracy.append(val_accuracy)
            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            # raise NotImplementedError()
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: \\ Done
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        C_img, H, W = img_shape

        # Weights shape: (D, n_classes)
        weights = self.weights

        # Remove bias (first feature)
        if has_bias:
            weights = weights[1:, :]        # -> (D-1, n_classes)

        n_classes = weights.shape[1]

        # Reshape directly
        w_images = weights.t().reshape(n_classes, C_img, H, W)

        return w_images


def hyperparams():
    hp = dict(weight_std=0.01, learn_rate=0.01, weight_decay=0.01)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.

    return hp
