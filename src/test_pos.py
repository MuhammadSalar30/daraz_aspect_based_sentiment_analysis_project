from src.pos_ner import CustomPOSTagger, CustomNERTagger

# 1. Create a tiny mathematical training set for the POS Tagger
# NN = Noun, ADJ = Adjective, VB = Verb, PREP = Preposition
training_data = [
    [("screen", "NN"), ("bohat", "ADJ"), ("achi", "ADJ"), ("hai", "VB")],
    [("battery", "NN"), ("kharab", "ADJ"), ("thi", "VB")],
    [("delivery", "NN"), ("jaldi", "ADJ"), ("hui", "VB")]
]

# 2. Train the POS Model
pos_model = CustomPOSTagger()
pos_model.fit(training_data)

# 3. Test data (Notice "samsung" and "karachi" are not in the POS training data)
test_sentence = ["Samsung", "ki", "screen", "achi", "hai", "daraz", "karachi", "se"]

# 4. Run POS Prediction
pos_tags = pos_model.predict(test_sentence)
print("--- POS TAGGING RESULTS ---")
print(pos_tags)

# 5. Run NER Extraction
ner_model = CustomNERTagger()
ner_tags = ner_model.extract_entities(test_sentence)
print("\n--- NER EXTRACTION RESULTS ---")
print(ner_tags)