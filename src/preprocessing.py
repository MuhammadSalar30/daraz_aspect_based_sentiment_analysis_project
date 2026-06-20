import csv
import re
from difflib import get_close_matches

# ─────────────────────────────────────────────────────────────────
#  Roman-Urdu spelling normalisation map
#  Covers the most common phonetic / typo variants seen in Daraz
#  reviews so that CRF aspect lookup always hits the canonical form.
# ─────────────────────────────────────────────────────────────────
SPELLING_VARIANTS = {
    # delivery variants
    "delievery": "delivery", "delivary": "delivery", "dilevery": "delivery",
    "dilvery":   "delivery", "delivry":  "delivery", "delievry": "delivery",
    # quality variants
    "qualtiy":  "quality",  "qualiy":   "quality",  "quilty":   "quality",
    "qulity":   "quality",  "qaulty":   "quality",
    # packaging variants
    "pakcing":  "packing",  "packging": "packing",  "pakaging": "packaging",
    # price / qeemat variants
    "qeemath":  "qeemat",   "qimat":    "qeemat",   "qemat":    "qeemat",
    # camera variants
    "camra":    "camera",   "cemara":   "camera",   "kamera":   "camera",
    # battery variants
    "battry":   "battery",  "batery":   "battery",  "battey":   "battery",
    # screen variants
    "scren":    "screen",   "screem":   "screen",
    # charger variants
    "charjer":  "charger",  "chargr":   "charger",
    # material variants
    "matrial":  "material", "materail": "material",
    # colour variants
    "colur":    "colour",   "colar":    "colour",   "collor":   "colour",
}


class DarazPreprocessor:
    def __init__(self):
        # Matches anything that is NOT alphanumeric or whitespace
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s]')

        # Build a fast-lookup set of known canonical words for fuzzy fallback
        self._canonical_words: set = set(SPELLING_VARIANTS.values())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def clean_text(self, text: str) -> str:
        """
        Converts text to lowercase and strips punctuation / special characters.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self.clean_pattern.sub('', text)
        return text

    def normalize_token(self, token: str) -> str:
        """
        Two-stage spelling normaliser:

        Stage 1 — Dictionary lookup
            Instantly corrects the most common Daraz / Roman-Urdu typos
            using the hand-crafted SPELLING_VARIANTS map (O(1)).

        Stage 2 — Fuzzy fallback  (difflib)
            If the token is not in the dictionary and is longer than 4 chars,
            attempt to match it against all known canonical forms.
            cutoff=0.82 keeps false-positive corrections very low.
        """
        lower = token.lower()

        # Stage 1: fast dictionary hit
        if lower in SPELLING_VARIANTS:
            return SPELLING_VARIANTS[lower]

        # Stage 2: fuzzy match against canonical words (only for longer tokens
        # to avoid mis-correcting short, legitimately ambiguous words)
        if len(lower) > 4:
            matches = get_close_matches(
                lower,
                self._canonical_words,
                n=1,
                cutoff=0.82
            )
            if matches:
                return matches[0]

        return token  # nothing matched — return as-is

    def tokenize(self, text: str) -> list:
        """
        Splits a normalised string into word tokens, applying spelling
        normalisation to every token before returning.
        """
        raw_tokens = [t for t in text.split(' ') if t]
        return [self.normalize_token(t) for t in raw_tokens]

    def load_and_filter_dataset(self, filepath: str) -> list:
        """
        Loads the CSV dataset, drops spam rows, and returns a list of dicts
        with normalised tokens and metadata.
        """
        processed_reviews = []

        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Drop spam rows (Dataset 2 only)
                if 'Spam' in row and row['Spam'].strip() == '1':
                    continue

                raw_review    = row.get('Reviews', '')
                raw_sentiment = row.get('Sentiment') or row.get('Sentiments', 'Neutral')
                cleaned       = self.clean_text(raw_review)
                tokens        = self.tokenize(cleaned)   # normalisation applied here

                processed_reviews.append({
                    'tokens':    tokens,
                    'rating':    row.get('Rating', None),
                    'sentiment': raw_sentiment,
                    'features':  row.get('Features', '')
                })

        return processed_reviews