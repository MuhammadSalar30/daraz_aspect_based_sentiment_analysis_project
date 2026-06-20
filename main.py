import os
import random
from collections import Counter

from src.preprocessing import DarazPreprocessor
from src.models_classical import CustomNaiveBayes, CustomLogisticRegression
from src.models_deep import CustomFFNN, CustomLSTM
from src.evaluation import Evaluator
from src.pos_ner import CustomNERTagger, CRFNERTagger
from src.corpus import BIO_TRAINING_DATA

CONTRAST_CONJUNCTIONS = {
    'lekin', 'magar', 'parantu', 'but', 'however',
    'albatta', 'phir bhi', 'mgr', 'lkn', 'par'
}


def _split_into_clauses(tokens: list) -> list:
    spans = []
    current_start = 0
    for i, token in enumerate(tokens):
        if token.lower() in CONTRAST_CONJUNCTIONS:
            if i > current_start:
                spans.append((current_start, i))
            current_start = i + 1
    if current_start < len(tokens):
        spans.append((current_start, len(tokens)))
    return spans if spans else [(0, len(tokens))]


def _build_clause_contexts(tokens: list, aspects: list, window_size: int) -> list:
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


def train_test_split_custom(data: list, test_ratio=0.2):
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    split_index = int(len(shuffled_data) * (1 - test_ratio))
    return shuffled_data[:split_index], shuffled_data[split_index:]


def balance_binary_dataset(X: list, y: list, seed=42) -> tuple:
    random.seed(seed)
    filtered = [(x, label) for x, label in zip(X, y)
                if str(label).capitalize() in ('Positive', 'Negative')]
    pos = [(x, l) for x, l in filtered if str(l).capitalize() == 'Positive']
    neg = [(x, l) for x, l in filtered if str(l).capitalize() == 'Negative']
    if not pos or not neg:
        return X, y
    min_count = min(len(pos), len(neg))
    balanced  = random.sample(pos, min_count) + random.sample(neg, min_count)
    random.shuffle(balanced)
    X_bal = [x for x, l in balanced]
    y_bal = [l for x, l in balanced]
    print(f"    [Balanced] Positive: {min_count}, Negative: {min_count}, "
          f"Total: {len(X_bal)}")
    return X_bal, y_bal


def predict_review(raw_text: str, model,
                   preprocessor: DarazPreprocessor,
                   crf_tagger: CRFNERTagger) -> None:
    print("\n" + "=" * 52)
    print(" ABSA Inference — CRF Sequence Extraction")
    print("=" * 52)
    print(f" Review : \"{raw_text}\"")
    print("-" * 52)

    cleaned = preprocessor.clean_text(raw_text)
    tokens  = preprocessor.tokenize(cleaned)

    if not tokens:
        print(" [!] Empty input.")
        print("=" * 52)
        return

    extracted_aspects = crf_tagger.extract_aspect_entities(tokens)
    contexts = _build_clause_contexts(tokens, extracted_aspects, window_size=5)

    if not contexts:
        overall = model.predict(tokens)
        icon    = "[+]" if overall == "Positive" else "[-]"
        print(" [!] No specific aspects detected.")
        print(f"     Fallback: {icon} {overall}")
        print("=" * 52)
        return

    results = []
    counts  = Counter()
    for ctx in contexts:
        sent = model.predict(ctx['context'])
        results.append((ctx['aspect'], sent, ctx['context']))
        counts[sent] += 1

    print(f" Detected {len(results)} aspect(s):\n")
    for aspect, sent, window in results:
        icon = "[+]" if sent == "Positive" else "[-]"
        print(f"  {icon} Aspect    : \"{aspect}\"")
        print(f"      Sentiment : {sent}")
        print(f"      Context   : \"{' '.join(window)}\"")
        print()

    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    if pos > 0 and neg > 0:
        verdict = "Mixed"
    elif pos > neg:
        verdict = "Overall Positive"
    else:
        verdict = "Overall Negative"

    print("-" * 52)
    print(f" Summary : {pos} Positive, {neg} Negative")
    print(f" Verdict : {verdict}")
    print("=" * 52)


def main():
    print("========================================")
    print(" Daraz ABSA Pipeline Initializing...")
    print("========================================")

    preprocessor = DarazPreprocessor()
    evaluator    = Evaluator(target_class="Positive")

    print("Training Linear-Chain CRF NER Engine...")
    crf_tagger = CRFNERTagger(epochs=15, learning_rate=0.1)
    crf_tagger.fit(BIO_TRAINING_DATA)

    dataset1_path = os.path.join("data", "raw", "daraz-dataset1.csv")
    dataset2_path = os.path.join("data", "raw", "daraz-dataset2.csv")
    processed_data_1 = []
    processed_data_2 = []

    if not os.path.exists(dataset1_path):
        print("\n[Error] Dataset not found. Using synthetic data.\n")
        raw_reviews = [
            ("The battery is absolutely excellent and lasts all day", "Positive"),
            ("Terrible delivery, the screen was completely broken",   "Negative"),
            ("Good price but the camera quality is very poor",        "Negative"),
            ("Fast delivery and amazing screen quality",              "Positive"),
        ]
        for text, sentiment in raw_reviews:
            tokens = preprocessor.tokenize(preprocessor.clean_text(text))
            processed_data_1.append({'tokens': tokens, 'sentiment': sentiment})
    else:
        print("Loading Primary Training Data...")
        processed_data_1 = preprocessor.load_and_filter_dataset(dataset1_path)
        if os.path.exists(dataset2_path):
            print("Loading Code-Mixed Stress Test Data...")
            processed_data_2 = preprocessor.load_and_filter_dataset(dataset2_path)
        else:
            print("[Warning] Code-Mixed data not found.")

    print("\nExecuting CRF Context Extraction...")
    extracted_dataset_1 = []
    extracted_dataset_2 = []
    discovered_aspects  = set()
    window_size         = 5

    for item in processed_data_1:
        tokens  = item['tokens']
        aspects = crf_tagger.extract_aspect_entities(tokens)
        discovered_aspects.update(aspects)
        for ctx in _build_clause_contexts(tokens, aspects, window_size):
            extracted_dataset_1.append({
                'context_window': ctx['context'],
                'sentiment':      item.get('sentiment', 'Neutral'),
            })

    for item in processed_data_2:
        tokens  = item['tokens']
        aspects = crf_tagger.extract_aspect_entities(tokens)
        for ctx in _build_clause_contexts(tokens, aspects, window_size):
            extracted_dataset_2.append({
                'context_window': ctx['context'],
                'sentiment':      item.get('sentiment', 'Neutral'),
            })

    print(f"--> CRF Verified Aspects: {discovered_aspects}\n")

    train_data, test_data = train_test_split_custom(extracted_dataset_1, test_ratio=0.2)
    X_train  = [d['context_window'] for d in train_data]
    y_train  = [d['sentiment']      for d in train_data]
    X_test   = [d['context_window'] for d in test_data]
    y_test   = [d['sentiment']      for d in test_data]
    X_stress = [d['context_window'] for d in extracted_dataset_2]
    y_stress = [d['sentiment']      for d in extracted_dataset_2]

    print(f"Dataset split : {len(X_train)} train, {len(X_test)} test windows.")
    print(f"Train labels  : {dict(Counter(y_train))}")
    print(f"Test labels   : {dict(Counter(y_test))}")
    if X_stress:
        print(f"Stress test   : {len(X_stress)} code-mixed windows.\n")

    classical_models = {
        "Naive Bayes":         CustomNaiveBayes(),
        "Logistic Regression": CustomLogisticRegression(learning_rate=0.01, epochs=20),
    }
    deep_models = {
        "Feed Forward NN": CustomFFNN(embedding_dim=16, hidden_dim=8,
                                      learning_rate=0.005, epochs=30),
        "LSTM":            CustomLSTM(embedding_dim=16, hidden_dim=8,
                                      learning_rate=0.005, epochs=25),
    }

    trained_models = {}
    X_bal, y_bal   = balance_binary_dataset(X_train, y_train)

    for model_name, model in classical_models.items():
        print(f"\n---> Training {model_name}...")
        model.train(X_bal, y_bal)
        trained_models[model_name] = model
        preds = [model.predict(w) for w in X_test]
        evaluator.display_report(f"{model_name} [Standard Test]", y_test, preds)
        if X_stress:
            preds_s = [model.predict(w) for w in X_stress]
            evaluator.display_report(
                f"{model_name} [Code-Mixed Stress Test]", y_stress, preds_s)
   

    for model_name, model in deep_models.items():
        print(f"\n---> Training {model_name}...")
        model.train(X_bal, y_bal)
        trained_models[model_name] = model
        preds = [model.predict(w) for w in X_test]
        evaluator.display_report(f"{model_name} [Standard Test]", y_test, preds)
        if X_stress:
            preds_s = [model.predict(w) for w in X_stress]
            evaluator.display_report(
                f"{model_name} [Code-Mixed Stress Test]", y_stress, preds_s)

    print("\n\n========================================")
    print(" Pipeline Training Complete.")
    print(" Best Model: Logistic Regression")
    print("========================================")

    best_model = trained_models["Logistic Regression"]

    # Demo predictions in terminal
    demo_reviews = [
        "battery bohat achi hai aur size bhi theek hai",
        "screen zabardast hai magar battery bekar hai",
    ]
    print("\n--- Demo Predictions ---")
    for review in demo_reviews:
        predict_review(review, best_model, preprocessor, crf_tagger)

if __name__ == "__main__":
    main()