import os
import random
from collections import Counter

from src.preprocessing import DarazPreprocessor
from src.models_classical import CustomNaiveBayes, CustomLogisticRegression
from src.models_deep import CustomFFNN, CustomLSTM
from src.evaluation import Evaluator
from src.pos_ner import CRFNERTagger
from src.corpus import BIO_TRAINING_DATA

CONTRAST_CONJUNCTIONS = {
    'lekin', 'magar', 'parantu', 'but', 'however',
    'albatta', 'phir bhi', 'mgr', 'lkn', 'par'
}

def _split_into_clauses(tokens):
    spans, current_start = [], 0
    for i, token in enumerate(tokens):
        if token.lower() in CONTRAST_CONJUNCTIONS:
            if i > current_start:
                spans.append((current_start, i))
            current_start = i + 1
    if current_start < len(tokens):
        spans.append((current_start, len(tokens)))
    return spans if spans else [(0, len(tokens))]

def _build_clause_contexts(tokens, aspects, window_size):
    clause_spans = _split_into_clauses(tokens)
    contexts = []
    for i, token in enumerate(tokens):
        if token.lower() not in [a.lower() for a in aspects]:
            continue
        clause_no = next(
            (idx for idx, (s, e) in enumerate(clause_spans) if s <= i < e),
            len(clause_spans) - 1
        )
        c_start, c_end = clause_spans[clause_no]
        clause_tokens  = tokens[c_start:c_end]
        local_idx      = i - c_start
        win_start = max(0, local_idx - window_size)
        win_end   = min(len(clause_tokens), local_idx + window_size + 1)
        contexts.append({
            'aspect':        token,
            'context':       clause_tokens[win_start:win_end],
            'clause_tokens': clause_tokens,
            'clause_index':  clause_no,
        })
    return contexts

def train_test_split_custom(data, test_ratio=0.2):
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]

def balance_binary_dataset(X, y, seed=42):
    random.seed(seed)
    filtered = [(x, l) for x, l in zip(X, y)
                if str(l).capitalize() in ('Positive', 'Negative')]
    pos = [(x, l) for x, l in filtered if str(l).capitalize() == 'Positive']
    neg = [(x, l) for x, l in filtered if str(l).capitalize() == 'Negative']
    if not pos or not neg:
        return X, y
    n = min(len(pos), len(neg))
    bal = random.sample(pos, n) + random.sample(neg, n)
    random.shuffle(bal)
    return [x for x, l in bal], [l for x, l in bal]

def main():
    preprocessor = DarazPreprocessor()
    evaluator    = Evaluator(target_class="Positive")

    crf_tagger = CRFNERTagger(epochs=15, learning_rate=0.1)
    crf_tagger.fit(BIO_TRAINING_DATA)

    dataset1_path = os.path.join("data", "raw", "daraz-dataset1.csv")
    dataset2_path = os.path.join("data", "raw", "daraz-dataset2.csv")

    processed_1 = []
    processed_2 = []

    if os.path.exists(dataset1_path):
        processed_1 = preprocessor.load_and_filter_dataset(dataset1_path)
    if os.path.exists(dataset2_path):
        processed_2 = preprocessor.load_and_filter_dataset(dataset2_path)

    extracted_1, extracted_2 = [], []
    window_size = 5

    for item in processed_1:
        tokens  = item['tokens']
        aspects = crf_tagger.extract_aspect_entities(tokens)
        for ctx in _build_clause_contexts(tokens, aspects, window_size):
            extracted_1.append({
                'context_window': ctx['context'],
                'sentiment':      item.get('sentiment', 'Neutral'),
            })

    for item in processed_2:
        tokens  = item['tokens']
        aspects = crf_tagger.extract_aspect_entities(tokens)
        for ctx in _build_clause_contexts(tokens, aspects, window_size):
            extracted_2.append({
                'context_window': ctx['context'],
                'sentiment':      item.get('sentiment', 'Neutral'),
            })

    train_data, test_data = train_test_split_custom(extracted_1, test_ratio=0.2)
    X_train = [d['context_window'] for d in train_data]
    y_train = [d['sentiment']      for d in train_data]
    X_test  = [d['context_window'] for d in test_data]
    y_test  = [d['sentiment']      for d in test_data]
    X_stress = [d['context_window'] for d in extracted_2]
    y_stress  = [d['sentiment']     for d in extracted_2]

    X_bal, y_bal = balance_binary_dataset(X_train, y_train)

    # Train Logistic Regression
    lr_model = CustomLogisticRegression(learning_rate=0.01, epochs=20)
    lr_model.train(X_bal, y_bal)
    evaluator.display_report("Logistic Regression [Standard Test]", y_test,
                             [lr_model.predict(w) for w in X_test])

    # ── DEBUG: inspect learned weights ───────────────────────────────────────
    vocab   = lr_model.vectorizer.vocabulary
    weights = lr_model.weights

    check_words = ['late', 'slow', 'weak', 'kharab', 'bekar', 'nahi',
                   'achi', 'zabardast', 'bohat', 'zyada', 'theek', 'sahi']

    print("\n" + "=" * 55)
    print(" DEBUG — Logistic Regression Learned Weights")
    print("=" * 55)
    for word in check_words:
        if word in vocab:
            idx = vocab[word]
            w   = weights.get(idx, 0.0)
            direction = "→ POSITIVE bias" if w > 0 else "→ NEGATIVE bias"
            print(f"  {word:<15} weight={w:+.5f}  {direction}")
        else:
            print(f"  {word:<15} NOT IN VOCABULARY (never seen in training)")
    print("=" * 55)

    # ── DEBUG: inspect specific context windows ───────────────────────────────
    print("\n DEBUG — Prediction breakdown for problem reviews:")
    problem_contexts = [
        ['charger', 'slow', 'hai'],
        ['delivery', 'bohat', 'late', 'thi'],
        ['delivery', 'late', 'thi'],
        ['quality', 'achi', 'hai'],
        ['battery', 'bekar', 'hai'],
    ]
    for ctx in problem_contexts:
        pred = lr_model.predict(ctx)
        x    = lr_model.vectorizer.transform_single(ctx)
        score = lr_model.bias + sum(
            lr_model.weights.get(fi, 0.0) * fw for fi, fw in x.items()
        )
        print(f"  {str(ctx):<45} → {pred}  (score={score:+.4f})")
    print()

if __name__ == "__main__":
    main()