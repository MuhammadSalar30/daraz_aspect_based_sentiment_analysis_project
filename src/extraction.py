from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  Contrastive conjunctions in Roman Urdu / English that signal a sentiment
#  REVERSAL between the left clause and the right clause.
# ─────────────────────────────────────────────────────────────────────────────
CONTRAST_CONJUNCTIONS = {
    'lekin', 'magar', 'parantu', 'but', 'however',
    'albatta', 'phir bhi', 'mgr', 'lkn', 'pr'
}


def _split_into_clauses(tokens: list) -> list[tuple[int, int]]:
    
    #Splits a token list into clause *spans* (start_idx, end_idx exclusive)on every contrastive conjunction.

    #Example:tokens = ['delivery', 'fast', 'thi', 'lekin', 'stand', 'toota', 'hua', 'tha']returns → [(0, 3), (4, 8)]          # 'lekin' at index 3 is the boundary
    
    spans   = []
    current_start = 0

    for i, token in enumerate(tokens):
        if token.lower() in CONTRAST_CONJUNCTIONS:
            if i > current_start:                      # non-empty left clause
                spans.append((current_start, i))
            current_start = i + 1                      # right clause starts after conjunction

    # Final (or only) clause
    if current_start < len(tokens):
        spans.append((current_start, len(tokens)))

    # If no conjunction was found, the whole sentence is one clause
    if not spans:
        spans = [(0, len(tokens))]

    return spans


def _clause_index_for(aspect_idx: int, clause_spans: list[tuple[int, int]]) -> int:
    """
    Returns the index (0-based) of the clause that contains `aspect_idx`.
    """
    for clause_no, (start, end) in enumerate(clause_spans):
        if start <= aspect_idx < end:
            return clause_no
    return len(clause_spans) - 1           # fallback: last clause


class POSAspectExtractor:

    #Discovers aspects via POS tagging and extracts *clause-bounded* context
    #windows for each detected aspect.



    def __init__(self, trained_pos_model, window_size: int = 3, top_n_aspects: int = 15):
        self.pos_model          = trained_pos_model
        self.window_size        = window_size
        self.top_n_aspects      = top_n_aspects
        self.discovered_aspects: set = set()


    def discover_aspects(self, dataset: list) -> set:
        
        #Dynamically discovers aspects by using the POS model's conditional probability to verify which words are Nouns (NN).
        
        word_frequencies: dict = defaultdict(int)

        for item in dataset:
            tokens = item.get('tokens', [])
            tagged_sentence = self.pos_model.predict(tokens)

            for word, tag in tagged_sentence:
                clean_word = word.lower().strip()
                if tag == "NN" and len(clean_word) > 2:
                    word_frequencies[clean_word] += 1

        sorted_aspects = sorted(
            word_frequencies.items(), key=lambda x: x[1], reverse=True
        )
        self.discovered_aspects = {
            word for word, _ in sorted_aspects[: self.top_n_aspects]
        }
        return self.discovered_aspects

    # ------------------------------------------------------------------ #
    #  Clause-aware context extraction  (core fix)                        #
    # ------------------------------------------------------------------ #

    def extract_aspect_contexts(self, tokens: list) -> list:
        """
        Slides a window across the sentence to isolate the context around
        each discovered aspect, but **constrains the window to the clause
        that contains the aspect**.

        Algorithm
        ---------
        1. Split the token list into clauses on contrastive conjunctions.
        2. For every token that matches a known aspect:
           a. Identify which clause it belongs to.
           b. Apply the sliding window *within that clause only*.
        3. Return one dict per aspect with keys:
              'aspect'        – the matched aspect token (original casing)
              'context'       – clause-bounded context window (list of tokens)
              'clause_index'  – which clause the aspect lives in (0-based)
              'clause_tokens' – the full clause the aspect belongs to
        """
        clause_spans  = _split_into_clauses(tokens)
        contexts      = []

        for i, token in enumerate(tokens):
            if token.lower() not in self.discovered_aspects:
                continue

            # ── 1. Find the clause this aspect belongs to ──────────────
            clause_no             = _clause_index_for(i, clause_spans)
            clause_start, clause_end = clause_spans[clause_no]
            clause_tokens         = tokens[clause_start:clause_end]

            # ── 2. Local index of the aspect *within* the clause ───────
            local_idx  = i - clause_start

            # ── 3. Apply window strictly inside the clause ─────────────
            win_start  = max(0,                  local_idx - self.window_size)
            win_end    = min(len(clause_tokens),  local_idx + self.window_size + 1)
            context_window = clause_tokens[win_start:win_end]

            contexts.append({
                'aspect':        token,
                'context':       context_window,
                'clause_index':  clause_no,
                'clause_tokens': clause_tokens,
            })

        return contexts