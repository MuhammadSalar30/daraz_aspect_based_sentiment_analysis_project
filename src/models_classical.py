import math
from collections import defaultdict
from src.features import CustomTFIDFVectorizer
class CustomNaiveBayes:
    def __init__(self):
        self.vocab = set()
        self.class_priors = {}
        self.word_likelihoods = defaultdict(lambda: defaultdict(float))
        self.class_counts = defaultdict(int)
        self.total_docs = 0

    def train(self, context_windows: list, labels: list):
        """
        Trains the Naive Bayes model by calculating class priors and
        word likelihoods using Maximum Likelihood Estimation with Laplace smoothing.
        
        Args:
            context_windows (list of lists): List containing tokenized text windows.
            labels (list): Matching list of sentiment labels (e.g., 'Positive', 'Negative').
        """
        self.total_docs = len(context_windows)
        if self.total_docs == 0:
            return

        # 1. Calculate class prior counts
        for label in labels:
            self.class_counts[label] += 1
            
        for label, count in self.class_counts.items():
            self.class_priors[label] = count / self.total_docs
            
        # 2. Count word frequencies per class
        word_counts_per_class = defaultdict(lambda: defaultdict(int))
        total_words_per_class = defaultdict(int)
        
        for tokens, label in zip(context_windows, labels):
            for token in tokens:
                self.vocab.add(token)
                word_counts_per_class[label][token] += 1
                total_words_per_class[label] += 1
                
        vocab_size = len(self.vocab)
        
        # 3. Calculate likelihoods P(w|c) using Laplace (Add-1) smoothing
        for label in self.class_counts.keys():
            denom = total_words_per_class[label] + vocab_size
            for word in self.vocab:
                num = word_counts_per_class[label][word] + 1
                self.word_likelihoods[label][word] = num / denom
                
            # Handle out-of-vocabulary (OOV) tokens dynamically using a safe baseline score
            self.word_likelihoods[label]['__UNSEEN__'] = 1 / denom

    def predict(self, context_window: list) -> str:
        """
        Predicts the class label for a single given context window using log probabilities
        to avoid numerical underflow errors.
        """
        best_class = None
        highest_score = float('-inf')
        
        # If the model hasn't been trained, default safely
        if not self.class_priors:
            return "Neutral"
            
        for label in self.class_priors.keys():
            # Initialize score with the log of the class prior probability
            score = math.log(self.class_priors[label])
            
            # Sum the log likelihoods of each token in the window
            for token in context_window:
                if token in self.vocab:
                    score += math.log(self.word_likelihoods[label][token])
                else:
                    score += math.log(self.word_likelihoods[label]['__UNSEEN__'])
                    
            if score > highest_score:
                highest_score = score
                best_class = label
                
        return best_class


class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # This will now store weights linked to integer feature IDs, not strings
        self.weights = defaultdict(float) 
        self.bias = 0.0
        # Initialize the custom vectorizer asset
        self.vectorizer = CustomTFIDFVectorizer()

    def _sigmoid(self, z: float) -> float:
        """
        Computes the safe activation limit values for the sigmoid function.
        """
        if z < -50:
            return 0.0
        if z > 50:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def train(self, context_windows: list, labels: list):
        """
        Trains the binary classifier using Stochastic Gradient Descent (SGD)
        over a custom calculated TF-IDF feature space.
        """
        # 1. Map string target labels to binary numeric values
        binary_labels = [1 if str(l).lower() in ['1', 'positive', 'pos'] else 0 for l in labels]
        
        # 2. Fit the TF-IDF vectorizer on the incoming training data
        self.vectorizer.fit(context_windows)
        self.weights[-1] = -1.0   # OOV negative words → negative signal
        self.weights[-2] = +1.0   # OOV positive words → positive signal

        # 3. Optimization Training Loop
        for epoch in range(self.epochs):
            for tokens, y in zip(context_windows, binary_labels):
                # Transform current token sequence into sparse TF-IDF weights {feature_idx: weight}
                x_tfidf = self.vectorizer.transform_single(tokens)
                
                # Compute linear matrix combination: z = sum(w * x) + b
                z = self.bias
                for feature_idx, tfidf_weight in x_tfidf.items():
                    z += self.weights[feature_idx] * tfidf_weight
                
                # Apply Sigmoid Activation Function
                y_hat = self._sigmoid(z)
                
                # Calculate Error Difference
                error = y_hat - y
                
                # Update Bias and Weights using gradients scaled by TF-IDF magnitude
                self.bias -= self.learning_rate * error
                for feature_idx, tfidf_weight in x_tfidf.items():
                    self.weights[feature_idx] -= self.learning_rate * error * tfidf_weight

    def predict(self, context_window: list) -> str:
        """
        Predicts sentiment classification based on calculated optimal feature thresholds.
        """
        # Transform unseen production inference text using the fitted IDF vocabulary
        x_tfidf = self.vectorizer.transform_single(context_window)
            
        z = self.bias
        for feature_idx, tfidf_weight in x_tfidf.items():
            z += self.weights[feature_idx] * tfidf_weight
            
        y_hat = self._sigmoid(z)
        return "Positive" if y_hat >= 0.5 else "Negative"