import abc
import torch
import random


class Layer(abc.ABC):
    """
    A Layer is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """

    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the layer.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the layer, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this layer.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Layer's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this layer between training and evaluation (test)
        mode. Some layers have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode

    def __repr__(self):
        return self.__class__.__name__


class LeakyReLU(Layer):
    """
    Leaky version of Rectified linear unit.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if not (0 <= alpha < 1):
            raise ValueError("Invalid value of alpha")
        self.alpha = alpha

    def forward(self, x, **kw):
        """
        Computes max(alpha*x, x) for some 0<= alpha < 1.
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        out = torch.where(x>0, x, self.alpha * x)
        
        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """
        x = self.grad_cache["x"]
        
        local_grad = torch.where(x>0, 1.0, self.alpha)
        
        dx = dout * local_grad
        
        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"LeakyReLU({self.alpha=})"


class ReLU(LeakyReLU):
    """
    Rectified linear unit.
    """

    def __init__(self):
        super().__init__(alpha=0.0)

    def __repr__(self):
        return "ReLU"


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes s(x) = 1/(1+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """
        
        out = 1.0 / (1.0 + torch.exp(-x))
        self.grad_cache["out"] = out

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """
        s_x = self.grad_cache["out"]
        local_grad = s_x * (1.0 - s_x)
        dx = dout * local_grad

        return dx

    def params(self):
        return []


class TanH(Layer):
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """
        out = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        self.grad_cache["out"] = out

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """
        
        tanh = self.grad_cache["out"]
        
        # 1. Calculate the local gradient: d(tanh(x))/dx = 1 - tanh^2(x) = 1 - y^2
        local_grad = 1.0 - tanh*tanh
        
        # 2. Apply the chain rule: dx = dout * local_grad
        dx = dout * local_grad

        return dx

    def params(self):
        return []


class Linear(Layer):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :param wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w = torch.normal(mean=0.0, std=wstd, size=(out_features, in_features)).float()
        self.b = torch.zeros(out_features).float()

        # These will store the gradients
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features.
        :return: Affine transform of each sample in x.
        """

        out = x @ self.w.t() + self.b

        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, Dout).
        :return: Gradient with respect to layer input, shape (N, Din)
        """
        x = self.grad_cache["x"]

        # 1. Gradient with respect to bias (db)
        # Sum dout over the batch dimension (N)
        db = dout.sum(dim=0)
        
        # 2. Gradient with respect to weights (dw)
        # dw = dout^T @ X
        # Since X is (N, Din) and dout is (N, Dout), we need dout.t() @ x
        dw = dout.t() @ x
        
        # 3. Gradient with respect to input (dx)
        # dx = dout @ W
        # Since dout is (N, Dout) and W is (Dout, Din), this is a direct matmul
        dx = dout @ self.w
        
        # ACCUMULATE gradients in self.dw and self.db
        self.dw += dw
        self.db += db

        return dx

    def __repr__(self):
        return f"Linear({self.in_features=}, {self.out_features=})"


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
            dimension, and D is the number of features. Should contain class
            scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
            each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
            scores, encoded y as 1-hot and calculated cross-entropy by
            definition above. A scalar.
        """

        # Shift input for numerical stability (already done by the provided code)
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax
        
        # 1. Calculate the log of the denominator term: log(sum_k e^(x_k))
        # This is log(sum(exp(x)))
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
        log_sum_exp = torch.log(sum_exp_x) # Shape (N, 1)

        # 2. Get the score of the correct class (x_y)
        # Use torch.gather to select the score x_y from each sample using the index y
        # y is shape (N,). Need to reshape y to (N, 1) for gather.
        x_y = torch.gather(x, dim=1, index=y.view(-1, 1)) # Shape (N, 1)

        # 3. Calculate the loss per sample: -x_y + log(sum_k e^(x_k))
        loss_per_sample = -x_y + log_sum_exp # Shape (N, 1)

        # 4. Final loss is the mean over the batch
        loss = torch.mean(loss_per_sample) # Scalar
        # ========================

        # Note: We save the numerically stable shifted input 'x' and labels 'y'
        # for the backward pass.
        self.grad_cache["x"] = x
        self.grad_cache["y"] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to layer output, a scalar which
            defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to layer input (only x), shape (N,D)
        """
        x = self.grad_cache["x"]
        y = self.grad_cache["y"]
        N = x.shape[0]
        D = x.shape[1]

        # 1. Calculate Softmax output (predicted probabilities, y_hat)
        # We use the shifted scores x saved from the forward pass
        exp_x = torch.exp(x)
        y_hat = exp_x / torch.sum(exp_x, dim=1, keepdim=True) # Shape (N, D)

        # 2. Create the one-hot encoding of the ground truth labels (y)
        # Shape (N, D). Uses y indices to place 1.0.
        y_one_hot = torch.zeros_like(y_hat).scatter_(1, y.view(-1, 1), 1.0)
        
        # 3. Calculate the derivative: dx = y_hat - y_one_hot
        dx = y_hat - y_one_hot
        
        # 4. Scale by the upstream gradient (dout) and normalize by batch size (N)
        # Note: Since the loss in forward was the MEAN, the backward gradient must be scaled by 1/N.
        dx = (dx / N) * dout
        # ========================

        return dx

    def params(self):
        return []


class Dropout(Layer):
    def __init__(self, p=0.5):
        """
        Initializes a Dropout layer.
        :param p: Probability to drop an activation.
        """
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x, **kw):
        # TODO: Implement the dropout forward pass.
        #  Notice that contrary to previous layers, this layer behaves
        #  differently a according to the current training_mode (train/test).
        # ====== YOUR CODE: ======
        
        if self.training_mode:
            # Zero in probability p
            mask = (torch.rand_like(x) > self.p).float()
            scale = 1.0 / (1.0 - self.p)
            # Keep mask and scale for backward pass
            self.grad_cache["mask"] = mask
            self.grad_cache["scale"] = scale

            out = x*mask*scale
        else:
            out = x
            
        # ========================

        return out

    def backward(self, dout):
        # TODO: Implement the dropout backward pass.
        # ====== YOUR CODE: ======

        if self.training_mode:
            mask = self.grad_cache["mask"]
            scale = self.grad_cache["scale"]

            dx = dout*mask*scale
        else:
            dx = dout

        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Layer):
    """
    A Layer that passes input through a sequence of other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, **kw):
        out = None
        pre_layer_out = x

        for layer in self.layers:
            out = layer(pre_layer_out, **kw)
            pre_layer_out = out

        return out

    def backward(self, dout):
        din = None
        # Start with the gradient coming into the last layer (dout)
        current_gradient = dout
        
        # Iterate through the layers in reverse order for backpropagation
        for layer in reversed(self.layers):
            # The backward function returns the gradient w.r.t. its input (dx),
            # which then becomes the upstream gradient (dout) for the layer before it.
            current_gradient = layer.backward(current_gradient)
        
        # The final gradient (din) is the gradient w.r.t. the overall input 'x'
        din = current_gradient
        # ========================

        return din

    def params(self):
        params = []

        # Collects all (parameter, gradient) tuples from every sub-layer
        for layer in self.layers:
            params.extend(layer.params())
        # ========================

        return params

    def train(self, training_mode=True):
        for layer in self.layers:
            layer.train(training_mode)

    def __repr__(self):
        res = "Sequential\n"
        for i, layer in enumerate(self.layers):
            res += f"\t[{i}] {layer}\n"
        return res

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers[item]


class MLP(Layer):
    """
    A simple multilayer perceptron based on our custom Layers.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(
        self,
        in_features,
        num_classes,
        hidden_features=(),
        activation="relu",
        dropout=0,
        **kw,
    ):
        super().__init__()
        """
        Create an MLP model Layer.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        layers = []

        # 1. Select the Activation Layer Class
        if activation == "relu":
            activation_cls = ReLU
        elif activation == "sigmoid":
            activation_cls = Sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Determine the full list of feature dimensions: [D, h1, h2, ..., hL, C]
        feature_dims = [in_features] + list(hidden_features) + [num_classes]

        # 2. Build the MLP sequence layer by layer
        # The loop iterates through the connections: (D, h1), (h1, h2), ..., (hL, C)
        
        # We stop at len(feature_dims) - 1 because we are looking at pairs (i, i+1)
        for i in range(len(feature_dims) - 1):
            dim_in = feature_dims[i]
            dim_out = feature_dims[i+1]
            
            # 2a. Add the Linear (FC) Layer
            # FC(Dim_in, Dim_out)
            layers.append(Linear(dim_in, dim_out, **kw))

            # 2b. Add Activation and Dropout (ONLY for hidden layers)
            # We skip adding activation/dropout after the very last Linear layer 
            # (which has num_classes output features)
            if i < len(feature_dims) - 2: 
                # Add the Activation Layer
                layers.append(activation_cls())
                
                # Add Dropout if p > 0
                if dropout > 0:
                    layers.append(Dropout(p=dropout))

        self.sequence = Sequential(*layers)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f"MLP, {self.sequence}"
