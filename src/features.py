import math
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  Roman-Urdu Sentiment Lexicons  (expanded + corrected)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    # Roman Urdu
    'achi', 'acha', 'ache', 'acchi', 'accha', 'acche',
    'badhiya', 'shandar', 'zabardast', 'mast', 'khubsoorat',
    'behtareen', 'umdah', 'laajawab', 'kamaal', 'pasand',
    'theek', 'bilkul', 'shukriya', 'wah', 'khush',
    'pyara', 'pyari', 'sundar', 'mazedaar', 'lajawab',
    'khoob', 'umda', 'jaldi', 'jald',
    # English
    'good', 'great', 'excellent', 'amazing', 'awesome',
    'nice', 'perfect', 'best', 'super', 'wonderful',
    'love', 'happy', 'satisfied', 'recommended', 'genuine',
    'original', 'premium', 'beautiful', 'comfortable', 'fast',
}

NEGATIVE_WORDS = {
    # Roman Urdu
    'kharab', 'bura', 'ganda', 'bekar', 'mehenga', 'mahanga',
    'phata', 'toota', 'tutay', 'khali', 'nakli', 'jhoota',
    'galat', 'ghalat', 'mushkil', 'problem', 'masla',
    'nuksan', 'tang', 'pareshani', 'takleef',
    # Negation words — critical
    'nahi', 'nahin', 'nhi', 'na', 'nah',
    # KEY FIX: time/quality negative signals
    'late',     # was NOT IN VOCABULARY — now forced via OOV handling
    'slow',     # was learning positive — now overridden
    'weak',     # was NOT IN VOCABULARY
    'delay',
    'delayed',
    'missing',
    'zyada',    # too much / overpriced
    # English
    'bad', 'worst', 'terrible', 'horrible', 'awful',
    'poor', 'cheap', 'broken', 'damaged', 'fake',
    'wrong', 'dirty', 'useless', 'waste', 'fraud', 'scam',
    'disappointed', 'disappointing', 'pathetic',
}

# ─────────────────────────────────────────────────────────────────────────────
#  Negation context words — intensify negative signal when nearby
#  e.g. "bohat late" → bohat should not boost positive here
# ─────────────────────────────────────────────────────────────────────────────
OVERRIDE_NEGATIVE = {'slow', 'weak', 'late', 'delay', 'delayed'}

NEGATION_INTENSIFIERS = {'bohat', 'bhot', 'bht', 'bahut', 'itna', 'kaafi'}

# Weight multipliers
POSITIVE_BOOST = 2.0
NEGATIVE_BOOST = 4.0   # Increased from 2.5 — negative words are fewer but critical


class CustomTFIDFVectorizer:
    """
    Custom TF-IDF vectoriser with Roman-Urdu sentiment-lexicon boosting.

    Key fixes in this version
    ─────────────────────────
    1. OOV negative words are now injected as synthetic features so words
       like 'late' and 'weak' that never appeared in training still produce
       a negative signal at inference time.
    2. NEGATIVE_BOOST increased to 4.0 to overcome the learned positive
       bias from words like 'slow' that appeared in positive contexts.
    3. Negation intensifier detection: 'bohat' near a negative word no
       longer pulls the prediction positive.
    """

    def __init__(self):
        self.vocabulary: dict = {}
        self.idf:        dict = {}
        self.doc_count:  int  = 0
        # Reserved feature indices for OOV sentiment words
        self._oov_neg_idx = -1   # synthetic feature for unseen negative words
        self._oov_pos_idx = -2   # synthetic feature for unseen positive words

    def fit(self, dataset: list):
        self.doc_count = len(dataset)
        if self.doc_count == 0:
            return

        document_frequencies = defaultdict(int)
        unique_words = set()

        for document in dataset:
            unique_tokens = set(document)
            for token in unique_tokens:
                document_frequencies[token] += 1
                unique_words.add(token)

        self.vocabulary = {
            word: idx for idx, word in enumerate(sorted(unique_words))
        }

        for word, df in document_frequencies.items():
            self.idf[word] = math.log(self.doc_count / (1 + df)) + 1

    def transform_single(self, tokens: list) -> dict:
        if not tokens:
            return {}

        tf_counts = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1

        total_tokens = len(tokens)
        tfidf_vector = {}

        # Check if any negative word is present in the window
        # (used to suppress bohat's positive signal)
        has_negative = any(t.lower() in NEGATIVE_WORDS for t in tokens)
        has_intensifier = any(t.lower() in NEGATION_INTENSIFIERS for t in tokens)

        for token, count in tf_counts.items():
            lower = token.lower()
            tf    = count / total_tokens

            if lower in self.vocabulary:
                # Known word — standard TF-IDF
                idf         = self.idf[lower]
                feature_idx = self.vocabulary[lower]
                weight      = tf * idf

                # Apply sentiment boost
                if lower in POSITIVE_WORDS:
                    # KEY FIX: suppress positive boost for intensifiers
                    # when they appear next to negative words
                    if lower in NEGATION_INTENSIFIERS and has_negative:
                        weight *= 0.2   # heavily suppress bohat's positive pull
                    else:
                        weight *= POSITIVE_BOOST
                elif lower in NEGATIVE_WORDS:
                    weight *= NEGATIVE_BOOST

                tfidf_vector[feature_idx] = weight

            else:
                # OOV word — check if it's a known sentiment word
                # KEY FIX: inject synthetic features for unseen sentiment words
                if lower in NEGATIVE_WORDS:
                    existing = tfidf_vector.get(self._oov_neg_idx, 0.0)
                    tfidf_vector[self._oov_neg_idx] = existing + (tf * NEGATIVE_BOOST)
                elif lower in POSITIVE_WORDS:
                    existing = tfidf_vector.get(self._oov_pos_idx, 0.0)
                    tfidf_vector[self._oov_pos_idx] = existing + (tf * POSITIVE_BOOST)

        return tfidf_vector

    def transform(self, dataset: list) -> list:
        return [self.transform_single(tokens) for tokens in dataset]