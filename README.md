Aspect Based Sentiment Analysis of Daraz Product Reviews
Project Overview
This repository contains a complete Aspect-Based Sentiment Analysis (ABSA) pipeline built specifically for code-mixed Roman Urdu e-commerce reviews (Daraz).Unlike standard NLP projects that rely on libraries like Scikit-Learn, PyTorch, or TensorFlow, the core Machine Learning models, mathematical vectorizers, and extraction algorithms in this project were built entirely from scratch using raw Python.
Key Features
Custom Conditional Random Field (CRF): A Linear-Chain CRF trained via Stochastic Gradient Ascent with Viterbi decoding to dynamically extract product aspects (e.g., "battery", "screen").
Clause-Bounded Context Windows: Syntactic splitting using contrastive conjunctions (e.g., 'lekin', 'magar') to prevent sentiment bleeding in multi-aspect reviews.
Lexicon-Boosted TF-IDF: A custom TF-IDF vectorizer that mathematically alters word weights based on hardcoded sentiment multipliers and negation intensifiers (handling the "bohat bekar" problem).
Deep & Classical ML from Scratch: Includes implementations of Naive Bayes, Logistic Regression, Feed-Forward Neural Networks (FFNN), and Long Short-Term Memory (LSTM) with custom Backpropagation Through Time (BPTT).
Interactive GUI: A fully functional Streamlit web application for real-time inference and model comparison.📂 Repository Structuredaraz-absa-project/
│
├── data/
│   └── raw/
│       ├── daraz-dataset1.csv       # Primary training data
│       └── daraz-dataset2.csv       # Code-mixed stress test data
│
├── src/
│   ├── corpus.py                    # BIO-tagged training data for CRF
│   ├── evaluation.py                # Custom metrics (F1, Precision, Recall)
│   ├── features.py                  # Custom TF-IDF & Lexicon Boost math
│   ├── models_classical.py          # Naive Bayes & Logistic Regression
│   ├── models_deep.py               # Custom FFNN & LSTM with BPTT
│   ├── pos_ner.py                   # CRF Tagger and Viterbi Decoder
│   └── preprocessing.py             # 2-Stage Spell Checking & Tokenization
│
├── app.py                           # Streamlit Web GUI
├── main.py                          # Terminal Orchestration & Training Pipeline
└── README.md
🔬 Scientific Conclusion: Deep Learning vs. Classical ML
This project served as a comparative study between classical ML and Deep Learning on highly noisy, code-mixed text.
Result: The custom Logistic Regression model drastically outperformed the custom LSTM.
Why? Deep learning models require massive, structured datasets to properly map dense multi-dimensional word embeddings. Because Roman Urdu is highly unstructured and our dataset was constrained, the LSTM overfit the data. Conversely, the sparse Bag-of-Words approach of TF-IDF, combined with targeted Lexicon Boosting, proved mathematically far more robust against local language noise.
How to Run Locally
Clone the repository
git clone https://github.com/your-username/daraz-absa-project.git
cd daraz-absa-project
Install dependencies(Note: Only standard libraries and Streamlit are required. No PyTorch/Scikit-Learn needed).
pip install streamlit
Run the Terminal Evaluation Pipeline
To train all 4 models and view the detailed evaluation metrics (Accuracy, Precision, Recall, F1, and Confusion Matrices
python main.py
Launch the Web InterfaceTo run the interactive Streamlit GUI for real-time review analysis:
streamlit run app.py
Developed as an NLP Academic Project.
