from torch.optim.lr_scheduler import MultiplicativeLR
from .utils import *
from collections import defaultdict
from itertools import combinations
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from random import sample
import gc
from time import time
from tqdm.auto import trange


default_device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseClassifier(torch.nn.Module):
    """
    Base class for all classifiers.
    """

    def __init__(self):

        super().__init__()

        self._termination = 0
        self._max_iter = 0

    def __activate__(self, loss_function: torch.nn.modules.loss, optimizer: torch.optim, optimizer_params: dict, lr_scheduler: torch.optim.lr_scheduler = MultiplicativeLR, scheduler_params: dict = {"lambda": lambda epoch: 1}, device: str = default_device):
        """
        Initializes loss function, optimizer and learning rate scheduler.

        :param loss_function: PyTorch loss function.
        :param optimizer: PyTorch optimizer.
        :param optimizer_params: Dictionary of parameters to be passed to the optimizer.
        :param lr_scheduler: PyTorch learning rate scheduler.
        :param scheduler_params: Dictionary of parameters to be passed to the learning rate scheduler.
        :param device: Device on which the computation should be performed ("cuda" or "cpu").
        """

        self.loss_function = loss_function
        self.optimizer = optimizer(self.parameters(), **optimizer_params)
        self.scheduler = lr_scheduler(self.optimizer, **scheduler_params)
        self.device = device



class FFNN(BaseClassifier):
    """
    Feed-Forward Neural Network classifier.
    """

    def __init__(self, input_size: int, hidden_sizes: Iterable[int], output_size: int, inner_activation: torch.nn.Module, final_activation: torch.nn.Module, optimizer: torch.optim, optimizer_params: dict, loss_function: torch.nn.modules.loss, lr_scheduler: torch.optim.lr_scheduler, scheduler_params: dict, device: str = default_device, dropout: float = 0):
        """

        :param input_size: Dimension of input vectors.
        :param hidden_sizes: Iterable of the dimensions of hidden layers.
        :param output_size: Dimension of the output.
        :param inner_activation: Activation function to be used in hidden layers.
        :param final_activation: Activation function to be used in the output layer.
        :param optimizer: PyTorch optimizer.
        :param optimizer_params: Dictionary of parameters to be passed to the optimizer.
        :param loss_function: PyTorch loss function.
        :param lr_scheduler: PyTorch learning rate scheduler.
        :param scheduler_params: Dictionary of parameters to be passed to the learning rate scheduler.
        :param device: Device on which the computation should be performed ("cuda" or "cpu").
        :param dropout: Fraction of neurons to be dropped at each training epoch.
        """

        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.validation_loss = []

        self.layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        layers = [
            [torch.nn.Linear(in_features = prev, out_features = post), inner_activation, torch.nn.Dropout(dropout)]
            for prev, post in zip(self.layer_sizes, self.layer_sizes[1:])]

        single_layers = sum(layers, [])[:-2]
        self.net = torch.nn.Sequential(*single_layers, final_activation)

        super().__activate__(loss_function, optimizer, optimizer_params, lr_scheduler, scheduler_params, device)


    def forward(self, X: torch.Tensor):

        X = X.float()
        Y = self.net(X)

        return Y


    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, classes: Iterable[Union[str, int]] = None, **kwargs):
        """
        Training step with partial data (X, y).

        :param X: Data features in tensor of shape (batch_size, self.input_size).
        :param y: Data labels in tensor of shape (batch_size, self.output_size).
        :param classes: Iterable of names of output classes (strings or integers), for compatibility with scikit-learn.
        """

        # Reshape y to output_size
        y = y.float().reshape((-1 ,self.output_size))

        # Define closure for LBFGS optimizer
        def closure():

            # Training step of PyTorch
            self.optimizer.zero_grad(set_to_none = True)
            y_pred = self.forward(X).float()
            try:
                loss = self.loss_function(y_pred, y)
            except:
                raise TypeError \
                    (f"Predictions have shape {y_pred.shape} and type {y_pred.dtype}, while ground truth has shape {y.shape} and type {y.dtype}")

            self.best_loss_ = loss.item()
            loss.backward()
            return loss

        self.optimizer.step(closure)


    def predict_proba(self, X, **kwargs):
        """
        Predict output probabilities for input vectors in tensor X.

        :param X: Tensor of shape (batch_size, self.input_size)
        :return: Tensor of shape (batch_size, self.output_size)
        """

        with torch.no_grad():
            Y_pred = self.forward(X)

        return Y_pred



class APPNP(BaseClassifier):
    """
    APPNP classifier, based on the paper "Predict then propagate: Graph neural networks meet personalized pagerank" [J. Gasteiger, A. Bojchevski, S. Gunnemann - 2018].
    """

    def __init__(self, input_size: int, output_hidden_size: int, output_size: int, depth: int, alphas: Union[float, List],
                 activation_function: torch.nn.modules.activation, loss_function: torch.nn.modules.loss,
                 optimizer: torch.optim, optimizer_params: dict, lr_scheduler: torch.optim.lr_scheduler, scheduler_params: dict, device: str = default_device, input_hidden_size: int = 0, dropout: float = 0):
        """

        :param input_size: Dimension of input node embeddings.
        :param output_hidden_size: Hidden dimension of the output dense classifier.
        :param output_size: Dimension of the output.
        :param depth: Number of iterations of APPNP.
        :param alphas: Weights in [0;1] of the original node vectors in each iteration of APPNP (if a single number is provided, then it is applied to all iterations).
        :param activation_function: Activation function.
        :param optimizer: PyTorch optimizer.
        :param optimizer_params: Dictionary of parameters to be passed to the optimizer.
        :param loss_function: PyTorch loss function.
        :param lr_scheduler: PyTorch learning rate scheduler.
        :param scheduler_params: Dictionary of parameters to be passed to the learning rate scheduler.
        :param device: Device on which the computation should be performed ("cuda" or "cpu").
        :param input_hidden_size: Dimension to map node embeddings before APPNP iterations (if = 0, keeps the original dimension, i.e. input_size).
        :param dropout: Fraction of neurons to be dropped at each training epoch.
        """

        super().__init__()

        self.input_size = input_size
        self.input_hidden_size = input_hidden_size if input_hidden_size else input_size
        self.output_hidden_size = output_hidden_size if output_hidden_size else output_size
        self.output_size = output_size
        self.dropout = dropout
        self.depth = depth
        self.alphas = alphas if isinstance(alphas, Iterable) else [alphas]*depth

        # Check depth >= 1
        assert self.depth >= 1

        # Check alphas in [0;1]
        assert all([a >= 0 and a <= 1 for a in alphas])

        # Check correct length of alphas
        assert len(self.alphas) == self.depth

        # Mapping of input node vectors
        self.input_layer = torch.nn.Linear(in_features=input_size, out_features=input_hidden_size) if input_hidden_size else torch.nn.Identity()

        # Output dense classifier
        self.output_layer = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=2*self.input_hidden_size, out_features=self.output_hidden_size),
            activation_function,
            torch.nn.Linear(in_features=self.output_hidden_size, out_features=output_size),
            torch.nn.Softmax(dim=-1)
        )

        super().__activate__(loss_function, optimizer, optimizer_params, lr_scheduler, scheduler_params, device)


    def forward(self, Ps: List[torch.Tensor], Xs: List[torch.Tensor]):

        # Pad the list of graph (adjacency or laplacian) matrices to the highest number of nodes (N_nodes_max)
        P_tilde = pad_sequence_2d(Ps)
        P_tilde = torch.stack(P_tilde).to(self.device)
        # P_tilde.shape = batch_size, N_nodes_max, N_nodes_max

        # Pad the list of node features of each graph
        X_tilde = torch.nn.utils.rnn.pad_sequence(Xs, batch_first=True).float().to(self.device)
        # X_tilde.shape = batch_size, N_nodes_max, self.input_size

        # Iterations
        H0 = self.input_layer(X_tilde)
        # H0.shape = batch_size, N_nodes_max, self.input_hidden_size

        H = torch.clone(H0.detach())
        for i in range(self.depth):
            # Matrix multiplication: H = (1 - a) * P @ H + a * H0
            H = (1 - self.alphas[i]) * torch.bmm(P_tilde, H) + self.alphas[i] * H0

        # Output
        Y = self.output_layer(H)
        # Y.shape = batch_size, N_nodes_max, self.output_size

        return Y


    def partial_fit(self, Ps: List[torch.Tensor], Xs: List[torch.Tensor], Y: List[List[Union[str, int, np.ndarray]]], classes: Iterable[Union[str, int]] = [], **kwargs):
        """
        Training step with partial data (P, X, y).

        :param Ps: List of (adjacency or laplacian) matrices of graphs.
        :param Xs: List of tensors containing node features for each graph.
        :param Y: List of labels or label probability arrays of each node in each graph.
        :param classes: Classes of the output.
        """

        lengths = list(map(len, Y))

        # Concatenate all (one-hot encoded) labels
        Y = sum(Y, [])
        if isinstance(Y[0], str):
            label_map = {s: i for i, s in enumerate(classes)}
            y_true = torch.from_numpy(np.array([label_map[s] for s in Y]))
        else:
            y_true = torch.from_numpy(np.array(Y))

        y_true = y_true.to(self.device)

        # Closure for LBFGS optimizer
        def closure():
            self.optimizer.zero_grad(set_to_none=True)

            y_pred = self.forward(Ps, Xs)

            # Truncate sequences to exclude padding nodes
            y_pred_trunc = torch.cat([y_pred_seq[:l] for y_pred_seq, l in zip(y_pred, lengths)])

            loss = self.loss_function(y_pred_trunc, y_true)
            self.best_loss_ = loss.item()
            loss.backward()

            return loss

        self.optimizer.step(closure)


    def predict_proba(self, Ps, Xs, **kwargs):
        """

        :param Ps: List of (adjacency or laplacian) matrices of graphs.
        :param Xs: List of tensors containing node features for each graph.
        :return: Tensor of shape (batch_size, N_nodes_max, output_size) containing the probabilities of each label for each node in each graph.
        """

        # For single instances
        if not isinstance(Ps, List):
            Ps = [Ps]
        if not isinstance(Xs, List):
            Xs = [Xs]

        with torch.no_grad():
            Y_pred = self.forward(Ps, Xs)

        return Y_pred



class GeneralCRF(BaseClassifier):
    """
    Conditional Random Field on graph of any structure.
    """

    def __init__(self, input_size: int, output_size: int, bias: bool, loss_function: torch.nn.modules.loss,
                 optimizer: torch.optim, optimizer_params: dict, lr_scheduler: torch.optim.lr_scheduler, scheduler_params: dict, device: str = default_device, dropout: float = 0, termination: float = 1e-2, max_iter: int = 100, delta_function: torch.nn.modules.loss = torch.nn.L1Loss()):
        """
        :param input_size: Dimension of input node embeddings.
        :param output_size: Dimension of the output.
        :param bias: Whether to compute bias term in linear layers.
        :param optimizer: PyTorch optimizer.
        :param optimizer_params: Dictionary of parameters to be passed to the optimizer.
        :param loss_function: PyTorch loss function.
        :param lr_scheduler: PyTorch learning rate scheduler.
        :param scheduler_params: Dictionary of parameters to be passed to the learning rate scheduler.
        :param device: Device on which the computation should be performed ("cuda" or "cpu").
        :param dropout: Fraction of neurons to be dropped at each training epoch.
        :param termination: In the prediction phase, L1 difference threshold between consecutive outputs to terminate the cycle.
        :param max_iter: In the prediction phase, maximum number of iterations.
        :param delta_function: In the prediction phase, the loss function to compute the difference between consecutive outputs of the iterative computation.
        """

        super().__init__()

        self._termination = termination
        self._max_iter = max_iter
        self._delta_function = delta_function

        self.input_size = input_size    # input_size = N_features
        self.output_size = output_size  # output_size = N_classes

        self.net = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features = input_size + output_size, out_features = output_size, bias = bias),
            torch.nn.Softmax(dim = -1)
        )

        super().__activate__(loss_function, optimizer, optimizer_params, lr_scheduler, scheduler_params, device)


    def forward(self, Ps: List[torch.Tensor], Xs: List[torch.Tensor], Y_curr: List[torch.Tensor]):

        # Pad the list of graph (adjacency or laplacian) matrices to the highest number of nodes (N_nodes_max)
        P_tilde = pad_sequence_2d(Ps)
        P_tilde = torch.stack(P_tilde).float().to(self.device)
        P_tilde = torch.nn.functional.normalize(P_tilde, p = 1, dim = 1)
        # P_tilde.shape = batch_size, N_nodes_max, N_nodes_max

        # Pad the list of node features of each graph
        X_tilde = torch.nn.utils.rnn.pad_sequence(Xs, batch_first=True).float().to(self.device)
        # X_tilde.shape = batch_size, N_nodes_max, input_size

        # Pad and reshape current Y if necessary
        if isinstance(Y_curr, List):
            Y_tilde = torch.nn.utils.rnn.pad_sequence(Y_curr, batch_first = True).to(self.device)
        elif isinstance(Y_curr, torch.Tensor):
            if len(Y_curr.shape) != 3:
                Y_tilde = Y_curr.unsqueeze(0).to(self.device)
            else:
                Y_tilde = Y_curr.to(self.device)
        else:
            raise TypeError("Y_curr is neither List nor torch.Tensor")
        # Y_tilde.shape = batch_size, N_nodes, output_size

        # Multiply each graph (laplacian or adjacency) matrix with label probabilities to propagate them
        T = torch.bmm(P_tilde, Y_tilde.float())
        # T.shape = batch_size, N_nodes_max, output_size

        # Concatenate propagated label probabilities with node features
        H = torch.cat([X_tilde, T], dim = -1)
        # H.shape = batch_size, N_nodes_max, input_size + output_size

        # Recompute node label probabilities
        Y_new = self.net(H)
        # Y_new.shape = batch_size, N_nodes_max, output_size

        return Y_new


    def partial_fit(self, Ps: List[torch.Tensor], Xs: List[torch.Tensor], Y: List[List[Union[str, int, np.ndarray]]], classes: Iterable[Union[str, int]] = [], **kwargs):
        """
        :param Ps: List of (adjacency or laplacian) matrices of graphs.
        :param Xs: List of tensors containing node features for each graph.
        :param Y: List of labels or label probability arrays of each node in each graph.
        :param classes: Classes of the output.
        """

        lengths = list(map(len, Y))

        # Concatenate all (one-hot encoded) labels
        Y = sum(Y, [])
        if isinstance(Y[0], str):
            label_map = {s: i for i, s in enumerate(classes)}
            y_true = torch.from_numpy(np.array([label_map[s] for s in Y]))
        else:
            y_true = torch.from_numpy(np.array(Y))

        y_true = y_true.to(self.device)

        # Initialize current labels
        Y_curr = kwargs['Y_curr']

        # Closure for LBFGS optimizer
        def closure():
            self.optimizer.zero_grad(set_to_none=True)

            y_pred = self.forward(Ps, Xs, Y_curr)

            # Truncate sequences to exclude padding nodes
            y_pred_trunc = torch.cat([y_pred_seq[:l] for y_pred_seq, l in zip(y_pred, lengths)])

            loss = self.loss_function(y_pred_trunc, y_true)
            self.best_loss_ = loss.item()
            loss.backward()

            return loss

        self.optimizer.step(closure)


    def predict_proba(self, Ps, Xs, **kwargs):
        """
        :param Ps: List of (adjacency or laplacian) matrices of graphs.
        :param Xs: List of tensors containing node features for each graph.
        :return: Tensor of shape (batch_size, N_nodes_max, output_size) containing the probabilities of each label for each node in each graph.
        """

        # For single instances
        if not isinstance(Ps, List):
            Ps = [Ps]
        if not isinstance(Xs, List):
            Xs = [Xs]

        # Initialize prior probabilities
        Y_curr = kwargs['Y_curr']

        with torch.no_grad():
            # Iterative prediction
            for t in range(self._max_iter):
                Y_pred = self.forward(Ps, Xs, Y_curr).detach()
                Y_curr_padded = torch.nn.utils.rnn.pad_sequence(Y_curr, batch_first = True).to(self.device)
                delta = self._delta_function(Y_pred, Y_curr_padded)
                if delta.item() < self._termination:
                    break
                Y_curr = [ys for ys in Y_pred]

            if t >= self._max_iter - 1:
                print("[GeneralCRF.predict_proba] Convergence not reached...")

        return Y_pred



class PairGraphSeg():
    """
    Overall model that rewires a linear-chain model by predicting new links and pruning others.
    """

    def __init__(self, P_classifier, N_classifier, classes: Iterable[Union[str, int]] = [], cutoff_prob_high: float = 0.5, cutoff_prob_low: float = 0.0, rounding: bool = True, use_laplacian: bool = False, device: str = default_device):
        """
        :param P_classifier: Model to predict probability of a link between a pair of nodes (must be trainable by a 'partial_fit' method and applicable by a 'predict_proba' method).
        :param N_classifier: Model to predict classes of nodes in a graph (must be trainable by a 'partial_fit' method and applicable by a 'predict_proba' method).
        :param classes: List of classes to be predicted.
        :param cutoff_prob_high: Probability (tau) over which a new edge is retained.
        :param cutoff_prob_low: Probability (rho) under which an old edge is cut.
        :param rounding: Whether to round weights of final edges to 1.
        :param use_laplacian: Whether to use the normalized laplacian matrix instead of the normalized adjacency matrix.
        :param device: Device on which the computation should be performed ("cuda" or "cpu").
        """

        self.P_classifier = P_classifier.to(device) if hasattr(P_classifier, "to") else P_classifier
        self.N_classifier = N_classifier.to(device) if hasattr(N_classifier, "to") else N_classifier

        self.classes_ = classes
        self.device = device

        self.tau = cutoff_prob_high
        self.rho = cutoff_prob_low
        self.rounding = rounding
        self.use_laplacian = use_laplacian

        self.P_losses = []
        self.N_losses = []

        self.time_performance = defaultdict(list)


    def fit(self, X: List[List[np.ndarray]], Y: List[List[Union[str, int, np.ndarray]]], pair_sample: int = 10000,
            shuffle: bool = False, n_epochs: int = 1, verbose: bool = False, doc_batch_size: int = 8,
            pair_batch_size: int = 128, random_state: int = 42, break_seq: int = 15, tol: float = 1e-4):
        """
        Fit the model.

        :param X: Batch of sequences of feature vectors (e.g. word embeddings, document embeddings, etc).
        :param Y: Batch of sequences of labels (strings or integers) or probability arrays.
        :param pair_sample: Number of node pairs to sample for training P_classifier, if lower than the amount of data. An equal number of positive (same class) and negative pairs will be sampled.
        :param shuffle: Whether to shuffle training sequences.
        :param n_epochs: Number of training epochs.
        :param verbose: Whether to print average epoch losses.
        :param doc_batch_size: Number of documents per batch.
        :param pair_batch_size: Number of node pairs per batch for P_classifier.
        :param random_state: Random seed.
        :param break_seq: Length of the sequence to be analyzed for quitting the training cycle.
        :param tol: Loss to be achieved for quitting the training cycle.
        """

        # Check X and Y are lists
        if not isinstance(X, List):
            X = X.tolist()

        if not isinstance(Y, List):
            Y = Y.tolist()

        # If classes were not specified, derive them from the data labels
        if not self.classes_:
            if isinstance(Y[0][0], Iterable):
                self.classes_ = list(range(len(Y[0][0])))
            else:
                self.classes_ = list(set(sum(Y, [])))

        # Shuffle sequences
        if shuffle:
            X, Y = list(zip(*sample(list(zip(X, Y)), len(X))))

        # Initializations
        P_break = False
        N_break = False
        current_predictions = []

        for epoch in trange(n_epochs):
            if verbose:
                print("---\n EPOCH:", epoch)

            P_epoch_losses = []
            N_epoch_losses = []

            # Generate batches
            batch_gen = batch_generator(list(zip(X, Y)), doc_batch_size)

            for b, batch in enumerate(batch_gen):

                # Batch is made of B sequences s1, ..., sB of variable length l1, ..., lB
                X_batch, Y_batch = list(zip(*batch))
                X_batch_pairs, Y_batch_pairs = [], []
                doc_pairs = {}

                # Check the batches are lists of lists

                # Pair preparation
                if verbose:
                    print("\tPair preparation")
                tic = time()
                for n, (x_seq, y_seq) in enumerate(zip(X_batch, Y_batch)):
                    idx = list(range(len(x_seq)))
                    doc_pairs[n] = list(combinations(idx, 2))
                    #nonconsecutive_pairs = list(filter(lambda tup: np.abs(tup[1] - tup[0]) != 1, doc_pairs[n]))
                    x_pairs = [np.concatenate((x_seq[i], x_seq[j])) for i, j in doc_pairs[n]]
                    y_matrix = linear_kernel(y_seq, y_seq, dense_output = False) * 2
                    y_pairs = [np.clip(y_matrix[i,j], 0, 1) for i, j in doc_pairs[n]]
                    
                    X_batch_pairs.extend(x_pairs)
                    Y_batch_pairs.extend(y_pairs)
                self.time_performance["Pair preparation"].append(time() - tic)

                # Pair sampling
                if verbose:
                    print("\tPair sampling")
                tic = time()
                if pair_sample < len(X_batch_pairs):
                    stratification = list(map(np.ceil, Y_batch_pairs))
                    X_pair_sample, Y_pair_sample = resample(X_batch_pairs,
                                                            Y_batch_pairs,
                                                            n_samples=pair_sample,
                                                            random_state=random_state,
                                                            stratify=stratification)
                else:
                    X_pair_sample = X_batch_pairs
                    Y_pair_sample = Y_batch_pairs
                self.time_performance["Pair sampling"].append(time() - tic)
                
                del X_batch_pairs
                del Y_batch_pairs

                # Fit Pair classifier
                if verbose:
                    print("\tFit Pair classifier")
                tic = time()
                minibatch_gen = batch_generator(list(zip(X_pair_sample, Y_pair_sample)), pair_batch_size)
                for minibatch in minibatch_gen:
                    X_pair_minibatch, Y_pair_minibatch = list(zip(*minibatch))
                    if isinstance(self.P_classifier, torch.nn.Module):
                        X_pair_minibatch = tensor_check(X_pair_minibatch).to(self.device)
                        Y_pair_minibatch = tensor_check(Y_pair_minibatch).to(self.device)
                    self.P_classifier.partial_fit(X_pair_minibatch, Y_pair_minibatch, classes=[1,0])
                self.time_performance["Fit Pair classifier"].append(time() - tic)

                P_epoch_losses.append(self.P_classifier.best_loss_)

                # Construction of adjacency matrices, laplacian matrices and node feature matrices
                if verbose:
                    print("\tConstruction of adjacency matrices and node feature matrices")
                tic = time()
                Ps = []
                Xs = []
                for n, (x_seq, y_seq) in enumerate(zip(X_batch, Y_batch)):
                    # Adjacency matrix
                    A = np.clip(linear_kernel(y_seq, y_seq)*2, 0, 1)
                    if self.rounding:
                        A = np.ceil(A)

                    # Inverse root degree array
                    if self.use_laplacian:
                        d = np.ceil(A).sum(axis = 1)
                        ird = np.power(d, -0.5)
                        D = np.diag(d)
                        IRD = np.diag(ird)

                        # Normalized Laplacian matrix: L = D^(-1/2) @ (D - A) @ D^(-1/2)
                        P = IRD @ (D - A) @ IRD
                    else:
                        P = A
                    P = torch.from_numpy(P)
                    Ps.append(P)

                    # Feature matrix X
                    X_feats = torch.from_numpy(np.stack(x_seq))
                    Xs.append(X_feats)

                self.time_performance["Graph construction"].append(time() - tic)

                # Initially fill current predictions with random probabilities (list of tensors of shapes of instances in Y_batch)
                if verbose:
                    print("\tRandom initialization of current predictions")
                if len(current_predictions) < b + 1:
                    tensor_shapes = [(len(a), len(self.classes_)) for a in Y_batch]
                    Y_curr = random_init_tensor_list(tensor_shapes)
                    current_predictions.append(Y_curr)

                # Fit node classifier and predict labels
                if verbose:
                    print("\tFit Node classifier")
                tic = time()
                self.N_classifier.partial_fit(Ps, Xs, Y_batch, classes=self.classes_, Y_curr = current_predictions[b])
                Y_new = self.N_classifier.predict_proba(Ps, Xs, Y_curr = current_predictions[b])
                self.time_performance["Fit Node classifier"].append(time() - tic)

                # Update current predictions for one batch at a time
                current_predictions[b] = [T for T in Y_new]

                N_epoch_losses.append(self.N_classifier.best_loss_)

            # Learning rate schedule update
            if hasattr(self.P_classifier, "scheduler"):
                self.P_classifier.scheduler.step()
            if hasattr(self.N_classifier, "scheduler"):
                self.N_classifier.scheduler.step()

            # Print epoch average losses if verbose
            if verbose:
                print("\tAvg. epoch P loss:", np.nanmean(P_epoch_losses))
                print("\tAvg. epoch N loss:", np.nanmean(N_epoch_losses))
            self.P_losses.append(np.nanmean(P_epoch_losses))
            self.N_losses.append(np.nanmean(N_epoch_losses))

            # Verify training cycle break conditions
            if epoch >= break_seq:
                P_break = train_break_rules(self.P_losses, tol, break_seq)
                N_break = train_break_rules(self.N_losses, tol, break_seq)

            if P_break and N_break:
                break

            gc.collect()

    def predict(self, X: List[List[np.ndarray]], priors: List[torch.Tensor] = [], Y_eval: List[List[Union[str, int, np.ndarray]]] = [], batch_pairs: int = 1024, self_loops: bool = True, verbose: bool = True):
        """
        :param X: List of data sequences.
        :param priors: List of tensors containing label probabilities for each node in a sequence.
        :param Y_eval: List of validation label sequences (strings or integers) or sequences of label probability arrays.
        :param verbose: Whether to print tests.
        :return: List of lists of arrays containing label probabilities for each node.
        """

        # Check X is a list
        if not isinstance(X, List):
            X = X.tolist()

        # Initialize probabilities as uniform if not provided
        lengths = list(map(len, X))
        if not priors:
            priors = init_label_probs(lengths, self.classes_)

        Ps = []
        Xs = []
        predictions = []

        for x_seq in X:
            idx = list(range(len(x_seq)))
            pairs = list(combinations(idx, 2))
            X_pairs = [np.concatenate((x_seq[i], x_seq[j])) for i, j in pairs]
            if verbose:
                print(f"[PairGraphSeg.predict] Number of pairs: {len(X_pairs)}")
            
            pair_batches = batch_generator(X_pairs, batch_pairs)
            pair_probs = []
            
            if verbose:
                print("[PairGraphSeg.predict] Pairs batched.")
                
            for X_pairs_batch in pair_batches:
                
                X_pairs_batch = tensor_check(X_pairs_batch).to(self.device)
                
                batch_probs = self.P_classifier.predict_proba(X_pairs_batch)
                
                # Detach and back to the CPU if using a PyTorch model
                if hasattr(pair_probs, "detach"):
                    batch_probs = batch_probs.detach().cpu()[:,0]
                else:
                    batch_probs = batch_probs[:,0]
                    
                pair_probs.append(batch_probs)
                
            pair_probs = np.concatenate(pair_probs) if isinstance(batch_probs, np.ndarray) else torch.cat(pair_probs)
            
            if verbose:
                print("[PairGraphSeg.predict] Pair probabilities computed.")

            # If an evaluation label sequence is given, compute the performance of P classifier alone and print evaluation
            if Y_eval:

                if isinstance(Y_eval[l][0], np.ndarray):
                    Y_pairs = [np.clip(2 * np.dot(Y_eval[l][i], Y_eval[l][j]), 0, 1) for i, j in pairs]
                elif isinstance(Y_eval[l][0], str):
                    Y_pairs = [int(Y_eval[l][i] in Y_eval[l][j] or Y_eval[l][j] in Y_eval[l][i]) for i, j in pairs]
                else:
                    Y_pairs = [int(Y_eval[l][i] == Y_eval[l][j]) for i, j in pairs]

                Y_pairs = torch.tensor(Y_pairs)
                evaloss = torch.nn.L1Loss()
                diff = evaloss(Y_pairs, torch.round(pair_probs))
                if verbose:
                    print("[P_classifier.predict_proba] Probability L1 error:", diff)
                self.P_classifier.validation_loss.append(diff)

            # Edges with probability below rho are cut
            # Edges with probability above tau are all retained (with weight 1 if rounding is active)
            # Edges with probability between rho and tau are retained only if linking consecutive sentences
            P = torch.eye(len(x_seq)) if self_loops else torch.zeros(len(x_seq), len(x_seq))
            for (i,j), p in zip(pairs, pair_probs.tolist()):
                P[i,j] = p
                
            if verbose:
                print("[P_classifier.predict_proba] P matrix of shape:", P.shape)
                
            P = filter_tensor_elements(P, upper_threshold = self.tau, lower_threshold = self.rho)
            if self.rounding:
                P[P >= self.tau] = 1
            Ps.append(P)

            # Node features
            X_tensor = torch.from_numpy(np.stack(x_seq))
            Xs.append(X_tensor)

            if verbose:
                print(f"[PairGraphSeg.predict] Generating graph with {torch.count_nonzero(P).item()} edges.")

        # Prediction
        y_pred = self.N_classifier.predict_proba(Ps, Xs, Y_curr = priors)

        # Truncate sequences to exclude padding nodes
        predictions = [[y_pred_seq[j].cpu().numpy() for j in range(l)] for y_pred_seq, l in zip(y_pred, lengths)]

        return predictions
