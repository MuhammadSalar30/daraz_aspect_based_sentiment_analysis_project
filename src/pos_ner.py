"""
src/pos_ner.py

Contains:
  - CustomPOSTagger   : Probabilistic POS tagger (unchanged)
  - CustomNERTagger   : Gazetteer + heuristic NER tagger (unchanged)
  - CRFNERTagger      : CRF-based NER trained from scratch using only
                        standard Python libraries (math, collections, random)
"""

import math
import random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM POS TAGGER  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CustomPOSTagger:
    def __init__(self):
        self.word_tag_counts   = defaultdict(lambda: defaultdict(int))
        self.word_totals       = defaultdict(int)
        self.tag_probabilities = defaultdict(dict)
        self.default_tag       = "NN"

        self.stop_words = {
            'is', 'the', 'and', 'but', 'to', 'was', 'were', 'it', 'this', 'that',
            'with', 'for', 'a', 'an', 'of', 'in', 'on', 'or', 'so', 'are', 'am',
            'hai', 'hain', 'aur', 'magar', 'lekin', 'ki', 'ka', 'ko', 'se', 'pe',
            'par', 'thi', 'tha', 'the', 'bhi', 'hi', 'ye', 'yeh', 'woh', 'wo',
            'mein', 'me', 'may', 'main', 'mujhe', 'mera', 'meri', 'us', 'uski',
            'un', 'iss', 'isay', 'liye', 'laye', 'sath', 'saath', 'aik', 'ek',
            'do', 'k', 'ke', 'ho', 'hon', 'nahi', 'nai', 'na', 'kya', 'kyun',
            'jab', 'kar',
            'nhi', 'ni', 'tou', 'toh', 'tu', 'phir', 'sirf', 'bas',
            'baad', 'pehle', 'abhi', 'ab', 'tab', 'jis', 'jaise', 'waisa',
            'bohat', 'bhot', 'bht', 'zyada', 'ziyada', 'thoda', 'bilkul',
            'kuch', 'sab', 'apne', 'apna', 'apni', 'iska', 'uska',
            'kyunkay', 'kyunke', 'isliye', 'warna', 'agar', 'tor',
            'terhan', 'waqai', 'hota', 'hoti', 'diya', 'liya', 'karne',
            'kia', 'mila', 'isse', 'koi', 'mere', 'usay', 'aap',
            'din', 'waqt', 'ghanta', 'minute', 'baar', 'dafa',
            'khush', 'asan', 'sahi', 'aram', 'rang', 'tor', 'bhot',
            'lakin', 'hota', 'diya', 'tou', 'sirf', 'kaafi',
        }

        self.sentiment_words = {
            'good', 'bad', 'excellent', 'terrible', 'late', 'fast',
            'amazing', 'poor', 'best', 'worst', 'awesome', 'dead', 'great',
            'very', 'too', 'much',
            'achi', 'acha', 'ache', 'achay', 'bekar', 'kharab', 'zabardast',
            'bakwas', 'bohat', 'bht', 'boht', 'munasib', 'sasta', 'mehnga',
            'original', 'nakli', 'asli', 'purana', 'naya', 'chota', 'bada',
            'halka', 'mazboot', 'kamzoor', 'behtareen', 'bura', 'ghalat',
            'jaldi', 'dheere', 'theek', 'sahi', 'seedha', 'ulta',
            'mushkil', 'aasan', 'pukhta', 'naram', 'sakht',
        }

        self.product_nouns = {
            'phone', 'mobile', 'laptop', 'tablet', 'watch', 'camera',
            'battery', 'charger', 'cable', 'screen', 'display', 'speaker',
            'earphones', 'headphones', 'airpods', 'keyboard', 'mouse',
            'cover', 'case', 'adapter', 'wifi', 'bluetooth',
            'shirt', 'jacket', 'dress', 'shoes', 'chappal', 'trouser',
            'pant', 'kurta', 'dupatta', 'fabric', 'material', 'stitching',
            'color', 'colour', 'fit', 'size', 'design',
            'delivery', 'packing', 'packaging', 'price', 'quality',
            'quantity', 'weight', 'smell', 'taste', 'sound', 'charge',
            'time', 'order', 'product', 'item', 'cheez', 'maal',
            'daraz', 'seller', 'shop', 'service', 'support',
            'qeemat', 'mayar', 'miyaar', 'masnoaat', 'istemaal',
        }

    def fit(self, tagged_training_data: list):
        for sentence in tagged_training_data:
            for word, tag in sentence:
                clean_word = word.lower().strip()
                self.word_tag_counts[clean_word][tag] += 1
                self.word_totals[clean_word] += 1

        for word, tag_counts in self.word_tag_counts.items():
            for tag, count in tag_counts.items():
                self.tag_probabilities[word][tag] = count / self.word_totals[word]

    def predict(self, tokens: list) -> list:
        tagged_sentence = []
        for token in tokens:
            clean_word = token.lower().strip()

            if clean_word in self.tag_probabilities:
                best_tag = max(self.tag_probabilities[clean_word],
                               key=self.tag_probabilities[clean_word].get)
            elif clean_word in self.stop_words:
                best_tag = "PREP"
            elif clean_word in self.sentiment_words:
                best_tag = "ADJ"
            elif clean_word in self.product_nouns:
                best_tag = "NN"
            else:
                best_tag = self.default_tag

            tagged_sentence.append((token, best_tag))
        return tagged_sentence


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM NER TAGGER  (unchanged — gazetteer baseline)
# ─────────────────────────────────────────────────────────────────────────────

class CustomNERTagger:
    def __init__(self):
        self.brand_entities = {
            'samsung', 'apple', 'infinix', 'tecno', 'xiaomi', 'redmi', 'oppo',
            'vivo', 'realme', 'huawei', 'nokia', 'dawlance', 'haier', 'pel'
        }
        self.location_entities = {
            'karachi', 'lahore', 'islamabad', 'rawalpindi', 'peshawar', 'quetta', 'multan'
        }
        self.platform_entities = {
            'daraz', 'tcs', 'leopard', 'trax'
        }

    def extract_entities(self, tokens: list) -> list:
        entities = []
        for token in tokens:
            clean_token = token.lower().strip()
            entity_label = "O"

            if clean_token in self.brand_entities:
                entity_label = "B-BRAND"
            elif clean_token in self.location_entities:
                entity_label = "B-LOC"
            elif clean_token in self.platform_entities:
                entity_label = "B-PLATFORM"
            elif token.istitle() and len(token) > 2:
                entity_label = "B-ENTITY"

            entities.append((token, entity_label))
        return entities


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRF NER TAGGER  (from scratch — standard Python only)
# ─────────────────────────────────────────────────────────────────────────────

class CRFNERTagger:
    """
    A Linear-Chain CRF for Named Entity Recognition, implemented entirely
    with standard Python libraries (math, random, collections).

    Label set (BIO scheme):
        O          – not an entity
        B-BRAND    – beginning of a brand name
        B-PLATFORM – beginning of a platform/courier name
        B-LOC      – beginning of a location
        B-PRODUCT  – beginning of a product-category noun
        B-ASPECT   – beginning of a product-attribute aspect

    How a CRF works (plain English):
    ─────────────────────────────────
    A standard classifier scores each token independently.
    A CRF scores entire *sequences* of labels at once, so it can learn
    "B-BRAND is almost never followed by B-LOC" as a global constraint.

    It does this via two sets of learned weights (θ):
      • Emission weights  w[label][feature] — how well a feature fits a label
      • Transition weights t[prev_label][cur_label] — how likely two labels
        appear back-to-back

    Training uses gradient ascent on the log-likelihood of the gold labels,
    approximated with the Viterbi forward-backward algorithm.
    """

    # ── Label inventory ──────────────────────────────────────────────────────
    LABELS = ["O", "B-BRAND", "B-PLATFORM", "B-LOC", "B-PRODUCT", "B-ASPECT"]

    # ── Gazetteers (used as *features*, not hard rules) ──────────────────────
    _BRANDS     = {'samsung','apple','infinix','tecno','xiaomi','redmi','oppo',
                   'vivo','realme','huawei','nokia','dawlance','haier','pel'}
    _PLATFORMS  = {'daraz','tcs','leopard','trax'}
    _LOCATIONS  = {'karachi','lahore','islamabad','rawalpindi','peshawar','quetta','multan'}
    _PRODUCTS   = {'phone','mobile','laptop','tablet','watch','camera','battery',
                   'charger','cable','screen','display','speaker','earphones',
                   'headphones','airpods','keyboard','mouse','cover','case',
                   'shirt','jacket','dress','shoes','chappal','trouser','pant',
                   'kurta','dupatta','fabric','material','stitching'}
    _ASPECTS    = {'delivery','packing','packaging','price','quality','quantity',
                   'weight','smell','taste','sound','color','colour','fit',
                   'size','design','service','support','qeemat','mayar'}

    def __init__(self, learning_rate: float = 0.1, epochs: int = 30,
                 l2_lambda: float = 0.01):
        """
        Parameters
        ──────────
        learning_rate : step size for gradient ascent
        epochs        : passes over the training corpus
        l2_lambda     : L2 regularisation strength (prevents overfitting)
        """
        self.lr        = learning_rate
        self.epochs    = epochs
        self.l2        = l2_lambda

        # weights[(label, feature_name)] = float
        self.emission_weights  : dict = defaultdict(float)
        # weights[(prev_label, cur_label)] = float
        self.transition_weights: dict = defaultdict(float)

        self._trained = False

    # ─────────────────────────────────────────────────────────────────────────
    # A. FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _token_features(self, tokens: list, i: int) -> list:
        """
        Extracts a list of string feature names for token at position i.
        Features capture the token itself, its neighbours, and
        domain-specific properties relevant to Roman Urdu e-commerce text.
        """
        word  = tokens[i].lower().strip()
        feats = []

        # ── Surface features ─────────────────────────────────────────────────
        feats.append(f"word={word}")
        feats.append(f"prefix2={word[:2]}")
        feats.append(f"prefix3={word[:3]}")
        feats.append(f"suffix2={word[-2:]}")
        feats.append(f"suffix3={word[-3:]}")
        feats.append(f"len={len(word)}")
        feats.append(f"is_title={tokens[i].istitle()}")
        feats.append(f"is_upper={tokens[i].isupper()}")
        feats.append(f"has_digit={'1' if any(c.isdigit() for c in word) else '0'}")

        # ── Gazetteer features ───────────────────────────────────────────────
        feats.append(f"in_brands={word in self._BRANDS}")
        feats.append(f"in_platforms={word in self._PLATFORMS}")
        feats.append(f"in_locations={word in self._LOCATIONS}")
        feats.append(f"in_products={word in self._PRODUCTS}")
        feats.append(f"in_aspects={word in self._ASPECTS}")

        # ── Context: previous token ──────────────────────────────────────────
        if i > 0:
            prev = tokens[i - 1].lower().strip()
            feats.append(f"prev_word={prev}")
            feats.append(f"prev_in_brands={prev in self._BRANDS}")
            feats.append(f"prev_in_products={prev in self._PRODUCTS}")
        else:
            feats.append("BOS")          # Beginning Of Sentence

        # ── Context: next token ──────────────────────────────────────────────
        if i < len(tokens) - 1:
            nxt = tokens[i + 1].lower().strip()
            feats.append(f"next_word={nxt}")
            feats.append(f"next_in_aspects={nxt in self._ASPECTS}")
        else:
            feats.append("EOS")          # End Of Sentence

        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # B. SCORE (unnormalised log-potential for one token + label)
    # ─────────────────────────────────────────────────────────────────────────

    def _emission_score(self, label: str, features: list) -> float:
        """Dot product of emission weights with the feature vector."""
        return sum(self.emission_weights[(label, f)] for f in features)

    def _transition_score(self, prev_label: str, cur_label: str) -> float:
        return self.transition_weights[(prev_label, cur_label)]

    # ─────────────────────────────────────────────────────────────────────────
    # C. VITERBI DECODE  (inference — finds the best label sequence)
    # ─────────────────────────────────────────────────────────────────────────

    def _viterbi(self, token_features: list) -> list:
        """
        Standard Viterbi algorithm over the label lattice.

        token_features : list of feature lists, one per token position

        Returns the most probable label sequence (list of strings).
        """
        n      = len(token_features)
        labels = self.LABELS

        # viterbi[i][label] = best log-score reaching label at position i
        viterbi  = [defaultdict(lambda: -math.inf) for _ in range(n)]
        backptr  = [defaultdict(str)               for _ in range(n)]

        # ── Initialise (position 0, no previous label) ───────────────────────
        for label in labels:
            viterbi[0][label] = self._emission_score(label, token_features[0])

        # ── Recursion ────────────────────────────────────────────────────────
        for i in range(1, n):
            for cur_label in labels:
                e_score = self._emission_score(cur_label, token_features[i])
                best_score = -math.inf
                best_prev  = labels[0]

                for prev_label in labels:
                    score = (viterbi[i - 1][prev_label]
                             + self._transition_score(prev_label, cur_label)
                             + e_score)
                    if score > best_score:
                        best_score = score
                        best_prev  = prev_label

                viterbi[i][cur_label] = best_score
                backptr[i][cur_label] = best_prev

        # ── Backtrace ────────────────────────────────────────────────────────
        best_last  = max(labels, key=lambda l: viterbi[n - 1][l])
        path       = [best_last]
        for i in range(n - 1, 0, -1):
            path.append(backptr[i][path[-1]])
        path.reverse()
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # D. FORWARD ALGORITHM  (computes log-partition / normalisation constant Z)
    # ─────────────────────────────────────────────────────────────────────────

    def _log_sum_exp(self, values: list) -> float:
        """Numerically stable log-sum-exp over a list of floats."""
        if not values:
            return -math.inf
        max_v = max(values)
        if max_v == -math.inf:
            return -math.inf
        return max_v + math.log(sum(math.exp(v - max_v) for v in values))

    def _forward(self, token_features: list) -> tuple:
        """
        Forward algorithm — returns (alpha table, log Z).

        alpha[i][label] = log p(all tokens 0..i, label_i = label)
        log Z           = log of the partition function (normaliser)
        """
        n      = len(token_features)
        labels = self.LABELS

        alpha = [defaultdict(lambda: -math.inf) for _ in range(n)]

        # Position 0
        for label in labels:
            alpha[0][label] = self._emission_score(label, token_features[0])

        # Positions 1 … n-1
        for i in range(1, n):
            e_scores = {l: self._emission_score(l, token_features[i])
                        for l in labels}
            for cur_label in labels:
                incoming = [
                    alpha[i - 1][prev]
                    + self._transition_score(prev, cur_label)
                    + e_scores[cur_label]
                    for prev in labels
                ]
                alpha[i][cur_label] = self._log_sum_exp(incoming)

        log_Z = self._log_sum_exp([alpha[n - 1][l] for l in labels])
        return alpha, log_Z

    # ─────────────────────────────────────────────────────────────────────────
    # E. BACKWARD ALGORITHM  (needed for marginal probabilities)
    # ─────────────────────────────────────────────────────────────────────────

    def _backward(self, token_features: list) -> list:
        """
        Backward algorithm — returns beta table.

        beta[i][label] = log p(tokens i+1 … n-1 | label_i = label)
        """
        n      = len(token_features)
        labels = self.LABELS

        beta = [defaultdict(lambda: -math.inf) for _ in range(n)]

        # Last position
        for label in labels:
            beta[n - 1][label] = 0.0   # log(1)

        # Positions n-2 … 0
        for i in range(n - 2, -1, -1):
            e_scores_next = {l: self._emission_score(l, token_features[i + 1])
                             for l in labels}
            for cur_label in labels:
                outgoing = [
                    self._transition_score(cur_label, nxt)
                    + e_scores_next[nxt]
                    + beta[i + 1][nxt]
                    for nxt in labels
                ]
                beta[i][cur_label] = self._log_sum_exp(outgoing)

        return beta

    # ─────────────────────────────────────────────────────────────────────────
    # F. GRADIENT COMPUTATION  (one training sentence)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_gradients(self, token_features: list, gold_labels: list):
        """
        Returns gradient dictionaries for emission and transition weights.

        CRF gradient (emission for label y at position i):
            ∂ log p(y*) / ∂w = Σ_i f(x,i) · [y*_i == y]   ← empirical count
                              − Σ_i f(x,i) · p(y_i=y | x)  ← model expectation
        """
        labels = self.LABELS
        n      = len(token_features)

        alpha, log_Z = self._forward(token_features)
        beta         = self._backward(token_features)

        # ── Marginals p(label_i = y | x) ─────────────────────────────────────
        # log_marginal[i][y] = alpha[i][y] + beta[i][y] - log_Z
        log_marginals = []
        for i in range(n):
            lm = {}
            for y in labels:
                lm[y] = alpha[i][y] + beta[i][y] - log_Z
            log_marginals.append(lm)

        # ── Transition marginals p(y_{i-1}, y_i | x) ─────────────────────────
        log_trans_marginals = []
        for i in range(1, n):
            tm = {}
            e_scores = {l: self._emission_score(l, token_features[i])
                        for l in labels}
            for prev in labels:
                for cur in labels:
                    log_p = (alpha[i - 1][prev]
                             + self._transition_score(prev, cur)
                             + e_scores[cur]
                             + beta[i][cur]
                             - log_Z)
                    tm[(prev, cur)] = log_p
            log_trans_marginals.append(tm)

        # ── Emission gradients ────────────────────────────────────────────────
        emit_grad = defaultdict(float)
        for i, (feats, gold) in enumerate(zip(token_features, gold_labels)):
            for f in feats:
                # +1 for gold label (empirical count)
                emit_grad[(gold, f)] += 1.0
                # subtract model expectation
                for y in labels:
                    p_y = math.exp(log_marginals[i][y])
                    emit_grad[(y, f)] -= p_y

        # ── Transition gradients ──────────────────────────────────────────────
        trans_grad = defaultdict(float)
        for i in range(1, n):
            prev_gold = gold_labels[i - 1]
            cur_gold  = gold_labels[i]
            # +1 for gold transition
            trans_grad[(prev_gold, cur_gold)] += 1.0
            # subtract model expectation
            for prev in labels:
                for cur in labels:
                    p_pair = math.exp(log_trans_marginals[i - 1][(prev, cur)])
                    trans_grad[(prev, cur)] -= p_pair

        return emit_grad, trans_grad

    # ─────────────────────────────────────────────────────────────────────────
    # G. TRAINING
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, training_sentences: list):
        """
        Trains the CRF using stochastic gradient ascent.

        Parameters
        ──────────
        training_sentences : list of (tokens, labels) tuples
            tokens : list[str]  — raw token strings
            labels : list[str]  — gold BIO label per token (same length)

        Example
        ───────
        >>> data = [
        ...     (["samsung", "ka", "phone", "acha", "hai"],
        ...      ["B-BRAND", "O", "B-PRODUCT", "O", "O"]),
        ...     (["delivery", "late", "thi"],
        ...      ["B-ASPECT", "O", "O"]),
        ... ]
        >>> crf = CRFNERTagger()
        >>> crf.fit(data)
        """
        print(f"  [CRF] Training on {len(training_sentences)} sentences "
              f"for {self.epochs} epochs …")

        random.seed(42)

        for epoch in range(self.epochs):
            random.shuffle(training_sentences)
            total_loss = 0.0

            for tokens, gold_labels in training_sentences:
                if not tokens:
                    continue

                # Skip sentences with length mismatch
                if len(tokens) != len(gold_labels):
                    continue

                # Extract features for each token
                token_feats = [self._token_features(tokens, i)
                               for i in range(len(tokens))]

                # Compute gradients
                emit_grad, trans_grad = self._compute_gradients(
                    token_feats, gold_labels
                )

                # ── Gradient ascent update (with L2 regularisation) ───────────
                for key, grad in emit_grad.items():
                    self.emission_weights[key] += (
                        self.lr * (grad - self.l2 * self.emission_weights[key])
                    )

                for key, grad in trans_grad.items():
                    self.transition_weights[key] += (
                        self.lr * (grad - self.l2 * self.transition_weights[key])
                    )

                # ── Approximate log-likelihood for monitoring ─────────────────
                _, log_Z = self._forward(token_feats)
                gold_score = 0.0
                for i, (feats, gold) in enumerate(zip(token_feats, gold_labels)):
                    gold_score += self._emission_score(gold, feats)
                    if i > 0:
                        gold_score += self._transition_score(gold_labels[i-1], gold)
                total_loss += gold_score - log_Z

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:>3}/{self.epochs}  "
                      f"log-likelihood = {total_loss:.2f}")

        self._trained = True
        print("  [CRF] Training complete.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # H. INFERENCE (public API)
    # ─────────────────────────────────────────────────────────────────────────

    def extract_entities(self, tokens: list) -> list:
        """
        Runs Viterbi decoding and returns a list of (token, label) pairs.

        Falls back to gazetteer-only labels if the model has not been trained.

        Parameters
        ──────────
        tokens : list[str] — raw token strings

        Returns
        ───────
        list of (token: str, label: str) tuples
            label ∈ {"O", "B-BRAND", "B-PLATFORM", "B-LOC",
                     "B-PRODUCT", "B-ASPECT"}
        """
        if not tokens:
            return []

        if not self._trained:
            # Graceful fallback: use gazetteers directly
            fallback = CustomNERTagger()
            return fallback.extract_entities(tokens)

        token_feats = [self._token_features(tokens, i)
                       for i in range(len(tokens))]
        predicted_labels = self._viterbi(token_feats)
        return list(zip(tokens, predicted_labels))

    def extract_aspect_entities(self, tokens: list) -> list:
        """
        Convenience method: returns only tokens labelled as B-ASPECT
        or B-PRODUCT (the entity types relevant to ABSA).

        Returns
        ───────
        list of str — aspect/product token strings
        """
        tagged = self.extract_entities(tokens)
        return [tok for tok, label in tagged
                if label in ("B-ASPECT", "B-PRODUCT")]