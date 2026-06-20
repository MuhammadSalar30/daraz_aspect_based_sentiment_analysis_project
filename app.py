import streamlit as st
import os
import random
from collections import Counter

from src.preprocessing import DarazPreprocessor
from src.pos_ner import CRFNERTagger
from src.corpus import BIO_TRAINING_DATA
from src.models_classical import CustomNaiveBayes, CustomLogisticRegression
from src.models_deep import CustomFFNN, CustomLSTM

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daraz ABSA System",
    page_icon="🛍️",
    layout="wide"
)

# ── Clause logic ──────────────────────────────────────────────────────────────
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
            'aspect':  token,
            'context': clause_tokens[win_start:win_end],
        })
    return contexts

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def compute_metrics(y_true, y_pred, target="Positive"):
    yt = [str(y).capitalize() for y in y_true]
    yp = [str(y).capitalize() for y in y_pred]
    tp = sum(1 for a, b in zip(yt, yp) if a == target and b == target)
    fp = sum(1 for a, b in zip(yt, yp) if a != target and b == target)
    fn = sum(1 for a, b in zip(yt, yp) if a == target and b != target)
    tn = sum(1 for a, b in zip(yt, yp) if a != target and b != target)
    acc  = (tp + tn) / len(yt) if yt else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4),
            "cm": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}}

# ── Cached pipeline — trains ALL 4 models ─────────────────────────────────────
@st.cache_resource(show_spinner="Training all models — please wait…")
def load_pipeline():
    preprocessor = DarazPreprocessor()

    crf_tagger = CRFNERTagger(epochs=15, learning_rate=0.1)
    crf_tagger.fit(BIO_TRAINING_DATA)

    dataset1_path = os.path.join("data", "raw", "daraz-dataset1.csv")
    dataset2_path = os.path.join("data", "raw", "daraz-dataset2.csv")

    processed_1 = preprocessor.load_and_filter_dataset(dataset1_path)
    processed_2 = (preprocessor.load_and_filter_dataset(dataset2_path)
                   if os.path.exists(dataset2_path) else [])

    extracted_1, extracted_2 = [], []
    window_size = 5

    for item in processed_1:
        tokens  = item['tokens']
        aspects = crf_tagger.extract_aspect_entities(tokens)
        for ctx in _build_clause_contexts(tokens, aspects, window_size):
            extracted_1.append({'context_window': ctx['context'],
                                 'sentiment': item.get('sentiment', 'Neutral')})

    for item in processed_2:
        tokens  = item['tokens']
        aspects = crf_tagger.extract_aspect_entities(tokens)
        for ctx in _build_clause_contexts(tokens, aspects, window_size):
            extracted_2.append({'context_window': ctx['context'],
                                 'sentiment': item.get('sentiment', 'Neutral')})

    train_data, test_data = train_test_split_custom(extracted_1, test_ratio=0.2)
    X_train = [d['context_window'] for d in train_data]
    y_train = [d['sentiment']      for d in train_data]
    X_test  = [d['context_window'] for d in test_data]
    y_test  = [d['sentiment']      for d in test_data]
    X_stress = [d['context_window'] for d in extracted_2]
    y_stress = [d['sentiment']      for d in extracted_2]

    X_bal, y_bal = balance_binary_dataset(X_train, y_train)

    # ── Train all 4 models ────────────────────────────────────────────────────
    models = {
        "Naive Bayes":         CustomNaiveBayes(),
        "Logistic Regression": CustomLogisticRegression(learning_rate=0.01, epochs=20),
        "Feed Forward NN":     CustomFFNN(embedding_dim=16, hidden_dim=8,
                                          learning_rate=0.005, epochs=30),
        "LSTM":                CustomLSTM(embedding_dim=16, hidden_dim=8,
                                          learning_rate=0.005, epochs=25),
    }

    # Classical models train on full balanced data
    # Deep models also train on balanced data (same as main.py)
    all_metrics = {}
    trained_models = {}

    for name, model in models.items():
        model.train(X_bal, y_bal)
        trained_models[name] = model

        preds_std = [model.predict(w) for w in X_test]
        std_metrics = compute_metrics(y_test, preds_std)

        stress_metrics = None
        if X_stress:
            preds_stress = [model.predict(w) for w in X_stress]
            stress_metrics = compute_metrics(y_stress, preds_stress)

        all_metrics[name] = {
            "standard": std_metrics,
            "stress":   stress_metrics,
        }

    dataset_info = {
        "train_size":  len(X_bal),
        "test_size":   len(X_test),
        "stress_size": len(X_stress),
        "train_dist":  dict(Counter(y_train)),
        "test_dist":   dict(Counter(y_test)),
    }

    # Best model for inference = Logistic Regression
    best_model = trained_models["Logistic Regression"]

    return preprocessor, crf_tagger, best_model, all_metrics, dataset_info


# ── Load everything ───────────────────────────────────────────────────────────
preprocessor, crf_tagger, best_model, all_metrics, dataset_info = load_pipeline()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛍️ Daraz Review Analyzer")
st.markdown("**Roman Urdu Aspect-Based Sentiment Analysis (ABSA)**  ·  CRF + Logistic Regression")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Inference Engine", "📊 Model Performance Report"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a Customer Review")
    st.caption("Supports Roman Urdu, English, and code-mixed text.")

    user_review = st.text_area(
        "Paste a Daraz customer review here:",
        height=110,
       
    )

    example_col1, example_col2, example_col3 = st.columns(3)
    with example_col1:
        if st.button("📋 Example 1", use_container_width=True):
            st.session_state["example"] = "quality achi hai lekin delivery bohat late thi"
    with example_col2:
        if st.button("📋 Example 2", use_container_width=True):
            st.session_state["example"] = "screen zabardast hai magar battery bekar hai"
    with example_col3:
        if st.button("📋 Example 3", use_container_width=True):
            st.session_state["example"] = "material bohat sasta tha aur stitching bhi kharab thi"

    # If example button was clicked, show pre-filled text
    if "example" in st.session_state and not user_review.strip():
        user_review = st.session_state["example"]
        st.info(f"Example loaded: *{user_review}*")

    analyze_clicked = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

    if analyze_clicked:
        review_text = user_review.strip()
        if not review_text:
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Running CRF extraction and sentiment prediction…"):
                cleaned = preprocessor.clean_text(review_text)
                tokens  = preprocessor.tokenize(cleaned)

            if not tokens:
                st.error("No valid tokens found.")
            else:
                extracted_aspects = crf_tagger.extract_aspect_entities(tokens)
                contexts = _build_clause_contexts(tokens, extracted_aspects, window_size=5)

                st.markdown("---")
                st.subheader("Analysis Results")

                if not contexts:
                    overall = best_model.predict(tokens)
                    st.info("No specific product aspects detected — using full-review fallback.")
                    if overall == "Positive":
                        st.success(f"**Overall Sentiment:** {overall} ✅")
                    else:
                        st.error(f"**Overall Sentiment:** {overall} ❌")
                else:
                    counts = Counter()
                    aspect_results = []
                    for ctx in contexts:
                        sent = best_model.predict(ctx['context'])
                        counts[sent] += 1
                        aspect_results.append({
                            "aspect":    ctx['aspect'],
                            "sentiment": sent,
                            "context":   " ".join(ctx['context']),
                        })

                    # Display aspect cards in columns
                    n_cols = min(len(aspect_results), 3)
                    cols   = st.columns(n_cols)
                    for i, r in enumerate(aspect_results):
                        with cols[i % n_cols]:
                            if r["sentiment"] == "Positive":
                                st.success(
                                    f"**{r['aspect'].capitalize()}**\n\n"
                                    f"Sentiment: Positive 🟢"
                                )
                            else:
                                st.error(
                                    f"**{r['aspect'].capitalize()}**\n\n"
                                    f"Sentiment: Negative 🔴"
                                )
                            st.caption(f"Context: \"{r['context']}\"")

                    # Verdict banner
                    st.markdown("---")
                    pos = counts.get("Positive", 0)
                    neg = counts.get("Negative", 0)
                    if pos > 0 and neg > 0:
                        st.warning(
                            f"**Final Verdict: Mixed Sentiment** "
                            f"— {pos} positive, {neg} negative aspect(s) ⚖️"
                        )
                    elif pos > neg:
                        st.success(
                            f"**Final Verdict: Overall Positive** "
                            f"— {pos} positive aspect(s) ✅"
                        )
                    else:
                        st.error(
                            f"**Final Verdict: Overall Negative** "
                            f"— {neg} negative aspect(s) ❌"
                        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Evaluation — All 4 Models")
    st.caption(
        "All models trained from scratch using standard Python only. "
    )

    # Dataset summary
    with st.expander("📂 Dataset Summary", expanded=True):
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Balanced Training Windows", dataset_info["train_size"])
        dc2.metric("Test Windows",              dataset_info["test_size"])
        dc3.metric("Stress Test Windows",       dataset_info["stress_size"])
        st.caption(f"Train label distribution: {dataset_info['train_dist']}")
        st.caption(f"Test label distribution:  {dataset_info['test_dist']}")

    st.markdown("---")

    MODEL_ORDER = ["Naive Bayes", "Logistic Regression", "Feed Forward NN", "LSTM"]
    MODEL_ICONS = {
        "Naive Bayes":         "📊",
        "Logistic Regression": "📈",
        "Feed Forward NN":     "🧠",
        "LSTM":                "🔁",
    }

    for model_name in MODEL_ORDER:
        metrics = all_metrics.get(model_name, {})
        std     = metrics.get("standard", {})
        stress  = metrics.get("stress",   {})

        icon = MODEL_ICONS.get(model_name, "")
        with st.expander(f"{icon} {model_name}", expanded=(model_name == "Logistic Regression")):

            col_std, col_str = st.columns(2)

            # Standard test
            with col_std:
                st.markdown("**Standard Test (Clean Data)**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy",  f"{std.get('accuracy',  0):.4f}")
                m2.metric("Precision", f"{std.get('precision', 0):.4f}")
                m3.metric("Recall",    f"{std.get('recall',    0):.4f}")
                m4.metric("F1-Score",  f"{std.get('f1',        0):.4f}")
                if "cm" in std:
                    cm = std["cm"]
                    st.code(
                        f"Confusion Matrix\n"
                        f"  TP: {cm['TP']:<6} FP: {cm['FP']}\n"
                        f"  FN: {cm['FN']:<6} TN: {cm['TN']}"
                    )

            # Stress test
            with col_str:
                st.markdown("**Code-Mixed Stress Test**")
                if stress:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Accuracy",  f"{stress.get('accuracy',  0):.4f}")
                    s2.metric("Precision", f"{stress.get('precision', 0):.4f}")
                    s3.metric("Recall",    f"{stress.get('recall',    0):.4f}")
                    s4.metric("F1-Score",  f"{stress.get('f1',        0):.4f}")
                    if "cm" in stress:
                        cm = stress["cm"]
                        st.code(
                            f"Confusion Matrix\n"
                            f"  TP: {cm['TP']:<6} FP: {cm['FP']}\n"
                            f"  FN: {cm['FN']:<6} TN: {cm['TN']}"
                        )
                else:
                    st.info("Stress test data not available.")

    # Summary comparison table
    st.markdown("---")
    st.subheader("📋 Summary Comparison")

    rows = []
    for name in MODEL_ORDER:
        std    = all_metrics.get(name, {}).get("standard", {})
        stress = all_metrics.get(name, {}).get("stress",   {})
        rows.append({
            "Model":              name,
            "Std Accuracy":       f"{std.get('accuracy',  0):.4f}",
            "Std F1":             f"{std.get('f1',        0):.4f}",
            "Stress Accuracy":    f"{stress.get('accuracy',  0):.4f}" if stress else "N/A",
            "Stress F1":          f"{stress.get('f1',        0):.4f}" if stress else "N/A",
        })

    st.table(rows)

    st.markdown("---")
