import math
import random
random.seed(42) 
from collections import defaultdict

# --- Helper Math Functions ---

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def matrix_vector_multiply(M, v):
    return [dot_product(row, v) for row in M]

def sigmoid(x):
    if x < -50: return 0.0
    if x > 50: return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x):
    if x > 20: return 1.0
    if x < -20: return -1.0
    return math.tanh(x)

def random_matrix(rows, cols, scale=None):
    # Xavier initialization: scale = sqrt(2 / (rows + cols))
    if scale is None:
        scale = math.sqrt(2.0 / (rows + cols))
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

def random_vector(dim, scale=None):
    if scale is None:
        scale = math.sqrt(2.0 / dim)
    return [random.uniform(-scale, scale) for _ in range(dim)]


class CustomFFNN:
    
    '''Feed-Forward Neural Network with:
    - Xavier weight initialization (prevents vanishing/exploding gradients)
    - Tanh hidden activation (avoids dead neuron problem of ReLU with small weights)
    - Single hidden layer with SGD backpropagation '''
    
    def __init__(self, embedding_dim=16, hidden_dim=8, learning_rate=0.005, epochs=30):
        random.seed(42)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.embeddings = {}

        # Xavier initialization ensures variance is preserved through layers
        self.W1 = random_matrix(hidden_dim, embedding_dim)
        self.b1 = [0.0] * hidden_dim  # Biases start at zero

        self.W2 = random_vector(hidden_dim)
        self.b2 = 0.0  # Output bias starts neutral

    def _get_embedding(self, word):
        if word not in self.embeddings:
            # Larger init scale so embeddings have meaningful signal from the start
            self.embeddings[word] = [random.uniform(-0.5, 0.5)
                                     for _ in range(self.embedding_dim)]
        return self.embeddings[word]

    def _average_embedding(self, tokens):
        if not tokens:
            return [0.0] * self.embedding_dim
        avg = [0.0] * self.embedding_dim
        for token in tokens:
            emb = self._get_embedding(token)
            for i in range(self.embedding_dim):
                avg[i] += emb[i]
        for i in range(self.embedding_dim):
            avg[i] /= len(tokens)
        return avg

    def train(self, context_windows: list, labels: list):
        binary_labels = [1 if str(l).lower() in ['1', 'positive', 'pos'] else 0 for l in labels]

        for epoch in range(self.epochs):
            for tokens, y in zip(context_windows, binary_labels):
                # --- Forward Pass ---
                x = self._average_embedding(tokens)

                # Hidden layer with tanh (avoids dead neuron problem)
                h_pre = matrix_vector_multiply(self.W1, x)
                h_pre = [h_pre[i] + self.b1[i] for i in range(self.hidden_dim)]
                h_act = [tanh(v) for v in h_pre]

                # Output layer with sigmoid
                z2 = dot_product(self.W2, h_act) + self.b2
                y_hat = sigmoid(z2)

                # --- Backward Pass ---
                loss_grad = y_hat - y  # dL/dz2

                # Output layer gradients
                dW2 = [loss_grad * a for a in h_act]
                db2 = loss_grad

                # Hidden layer gradients (tanh derivative: 1 - tanh^2)
                dh = [loss_grad * self.W2[i] * (1.0 - h_act[i] ** 2)
                      for i in range(self.hidden_dim)]

                # W1 and b1 gradients
                dW1 = [[dh[i] * x[j] for j in range(self.embedding_dim)]
                       for i in range(self.hidden_dim)]

                # Embedding gradient
                dx = [sum(dh[i] * self.W1[i][j] for i in range(self.hidden_dim))
                      for j in range(self.embedding_dim)]

                # --- Weight Updates ---
                self.b2 -= self.lr * db2
                for i in range(self.hidden_dim):
                    self.W2[i] -= self.lr * dW2[i]
                    self.b1[i] -= self.lr * dh[i]
                    for j in range(self.embedding_dim):
                        self.W1[i][j] -= self.lr * dW1[i][j]

                # Update embeddings
                if tokens:
                    for token in tokens:
                        emb = self.embeddings[token]
                        for j in range(self.embedding_dim):
                            emb[j] -= (self.lr * dx[j]) / len(tokens)

    def predict(self, context_window: list) -> str:
        x = self._average_embedding(context_window)
        h_pre = matrix_vector_multiply(self.W1, x)
        h_pre = [h_pre[i] + self.b1[i] for i in range(self.hidden_dim)]
        h_act = [tanh(v) for v in h_pre]
        z2 = dot_product(self.W2, h_act) + self.b2
        return "Positive" if sigmoid(z2) >= 0.5 else "Negative"


class CustomLSTM:
    


    def __init__(self, embedding_dim=16, hidden_dim=8, learning_rate=0.005, epochs=25):

        random.seed(42)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.embeddings = {}

        concat_dim = embedding_dim + hidden_dim

        # Xavier initialization for all gate matrices
        self.W_i = random_matrix(hidden_dim, concat_dim)
        self.W_f = random_matrix(hidden_dim, concat_dim)
        self.W_o = random_matrix(hidden_dim, concat_dim)
        self.W_c = random_matrix(hidden_dim, concat_dim)

        # Standard practice: forget gate bias = 1.0 to encourage remembering early
        self.b_i = [0.0] * hidden_dim
        self.b_f = [1.0] * hidden_dim  # Key: forget gate bias starts at 1
        self.b_o = [0.0] * hidden_dim
        self.b_c = [0.0] * hidden_dim

        # Output projection
        self.W_v = random_vector(hidden_dim)
        self.b_v = 0.0

    def _get_embedding(self, word):
        if word not in self.embeddings:
            self.embeddings[word] = [random.uniform(-0.5, 0.5)
                                     for _ in range(self.embedding_dim)]
        return self.embeddings[word]

    def _forward_step(self, x_t, h_prev, c_prev):
        concat = x_t + h_prev  # list concatenation

        raw_i = [dot_product(self.W_i[k], concat) + self.b_i[k] for k in range(self.hidden_dim)]
        raw_f = [dot_product(self.W_f[k], concat) + self.b_f[k] for k in range(self.hidden_dim)]
        raw_o = [dot_product(self.W_o[k], concat) + self.b_o[k] for k in range(self.hidden_dim)]
        raw_c = [dot_product(self.W_c[k], concat) + self.b_c[k] for k in range(self.hidden_dim)]

        i_t = [sigmoid(v) for v in raw_i]
        f_t = [sigmoid(v) for v in raw_f]
        o_t = [sigmoid(v) for v in raw_o]
        c_tilde = [tanh(v) for v in raw_c]

        c_t = [f_t[k] * c_prev[k] + i_t[k] * c_tilde[k] for k in range(self.hidden_dim)]
        tanh_c_t = [tanh(c_t[k]) for k in range(self.hidden_dim)]
        h_t = [o_t[k] * tanh_c_t[k] for k in range(self.hidden_dim)]

        return h_t, c_t, i_t, f_t, o_t, c_tilde, tanh_c_t, concat

    def train(self, context_windows: list, labels: list):
        binary_labels = [1 if str(l).lower() in ['1', 'positive', 'pos'] else 0 for l in labels]

        for epoch in range(self.epochs):
            for tokens, y in zip(context_windows, binary_labels):
                if not tokens:
                    continue

                # --- Forward pass: cache all states ---
                h_states = [[0.0] * self.hidden_dim]
                c_states = [[0.0] * self.hidden_dim]
                cache = []

                for token in tokens:
                    x_t = self._get_embedding(token)
                    h_t, c_t, i_t, f_t, o_t, c_tilde, tanh_c_t, concat = \
                        self._forward_step(x_t, h_states[-1], c_states[-1])
                    h_states.append(h_t)
                    c_states.append(c_t)
                    cache.append((x_t, h_states[-2], c_states[-2],
                                  i_t, f_t, o_t, c_tilde, tanh_c_t, concat))

                # --- Output layer ---
                h_final = h_states[-1]
                z = dot_product(self.W_v, h_final) + self.b_v
                y_hat = sigmoid(z)
                loss_grad = y_hat - y

                # Update output projection
                self.b_v -= self.lr * loss_grad
                dh_final = [0.0] * self.hidden_dim
                for i in range(self.hidden_dim):
                    self.W_v[i] -= self.lr * loss_grad * h_final[i]
                    dh_final[i] = loss_grad * self.W_v[i]

                # --- Truncated BPTT through all timesteps ---
                dc_next = [0.0] * self.hidden_dim
                dh_next = dh_final

                for t in range(len(cache) - 1, -1, -1):
                    x_t, h_prev, c_prev, i_t, f_t, o_t, c_tilde, tanh_c_t, concat = cache[t]

                    dh = dh_next

                    # Gate gradients
                    do = [dh[k] * tanh_c_t[k] for k in range(self.hidden_dim)]
                    dc = [dh[k] * o_t[k] * (1.0 - tanh_c_t[k] ** 2) + dc_next[k]
                          for k in range(self.hidden_dim)]
                    dc_tilde = [dc[k] * i_t[k] for k in range(self.hidden_dim)]
                    di = [dc[k] * c_tilde[k] for k in range(self.hidden_dim)]
                    df = [dc[k] * c_prev[k] for k in range(self.hidden_dim)]
                    dc_prev = [dc[k] * f_t[k] for k in range(self.hidden_dim)]

                    # Pre-activation gradients
                    d_raw_i = [di[k] * i_t[k] * (1.0 - i_t[k]) for k in range(self.hidden_dim)]
                    d_raw_f = [df[k] * f_t[k] * (1.0 - f_t[k]) for k in range(self.hidden_dim)]
                    d_raw_o = [do[k] * o_t[k] * (1.0 - o_t[k]) for k in range(self.hidden_dim)]
                    d_raw_c = [dc_tilde[k] * (1.0 - c_tilde[k] ** 2) for k in range(self.hidden_dim)]

                    # Update all gate weights and biases
                    for k in range(self.hidden_dim):
                        self.b_i[k] -= self.lr * d_raw_i[k]
                        self.b_f[k] -= self.lr * d_raw_f[k]
                        self.b_o[k] -= self.lr * d_raw_o[k]
                        self.b_c[k] -= self.lr * d_raw_c[k]
                        for j in range(len(concat)):
                            self.W_i[k][j] -= self.lr * d_raw_i[k] * concat[j]
                            self.W_f[k][j] -= self.lr * d_raw_f[k] * concat[j]
                            self.W_o[k][j] -= self.lr * d_raw_o[k] * concat[j]
                            self.W_c[k][j] -= self.lr * d_raw_c[k] * concat[j]

                    # Propagate to previous timestep
                    dc_next = dc_prev
                    dh_next = [0.0] * self.hidden_dim
                    for k in range(self.hidden_dim):
                        for h_idx in range(self.hidden_dim):
                            cidx = self.embedding_dim + h_idx
                            dh_next[h_idx] += (
                                d_raw_i[k] * self.W_i[k][cidx] +
                                d_raw_f[k] * self.W_f[k][cidx] +
                                d_raw_o[k] * self.W_o[k][cidx] +
                                d_raw_c[k] * self.W_c[k][cidx]
                            )

    def predict(self, context_window: list) -> str:
        if not context_window:
            return "Negative"
        h = [0.0] * self.hidden_dim
        c = [0.0] * self.hidden_dim
        for token in context_window:
            x_t = self._get_embedding(token)
            h, c, _, _, _, _, _, _ = self._forward_step(x_t, h, c)
        z = dot_product(self.W_v, h) + self.b_v
        return "Positive" if sigmoid(z) >= 0.5 else "Negative"