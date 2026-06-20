# src/corpus.py

# Grammar Tags: 
# NN = Noun (Aspects), VB = Verb, ADJ = Adjective, PREP = Preposition, 
# ADV = Adverb, CONJ = Conjunction, PRO = Pronoun

ROMAN_URDU_CORPUS = [
    [("phone", "NN"), ("ki", "PREP"), ("battery", "NN"), ("bohat", "ADV"), ("achi", "ADJ"), ("hai", "VB")],
    [("daraz", "NN"), ("se", "PREP"), ("order", "NN"), ("kiya", "VB"), ("tha", "VB"), ("lekin", "CONJ"), ("delivery", "NN"), ("late", "ADJ"), ("thi", "VB")],
    [("screen", "NN"), ("ka", "PREP"), ("size", "NN"), ("chota", "ADJ"), ("hai", "VB"), ("magar", "CONJ"), ("quality", "NN"), ("zabardast", "ADJ"), ("hai", "VB")],
    [("camera", "NN"), ("bilkul", "ADV"), ("bekar", "ADJ"), ("hai", "VB"), ("aur", "CONJ"), ("awaaz", "NN"), ("bhi", "ADV"), ("kharab", "ADJ"), ("hai", "VB")],
    [("samsung", "NN"), ("ka", "PREP"), ("mobile", "NN"), ("original", "ADJ"), ("nahi", "ADV"), ("tha", "VB")],
    [("qeemat", "NN"), ("munasib", "ADJ"), ("hai", "VB"), ("aur", "CONJ"), ("packing", "NN"), ("achi", "ADJ"), ("thi", "VB")],
    [("laptop", "NN"), ("ka", "PREP"), ("keyboard", "NN"), ("kaam", "VB"), ("nahi", "ADV"), ("karta", "VB")],
    [("mujhe", "PRO"), ("mera", "PRO"), ("parcel", "NN"), ("mil", "VB"), ("gaya", "VB"), ("hai", "VB")],
    [("is", "PRO"), ("ki", "PREP"), ("speed", "NN"), ("bohat", "ADV"), ("fast", "ADJ"), ("hai", "VB")],
    [("awaz", "NN"), ("bohat", "ADV"), ("halki", "ADJ"), ("hai", "VB"), ("headphones", "NN"), ("ki", "PREP")],
    [("material", "NN"), ("bohat", "ADV"), ("sasta", "ADJ"), ("lag", "VB"), ("raha", "VB"), ("hai", "VB")],
    [("seller", "NN"), ("ne", "PREP"), ("ghalat", "ADJ"), ("cheez", "NN"), ("bheji", "VB"), ("hai", "VB")],
    [("airpods", "NN"), ("ki", "PREP"), ("sound", "NN"), ("bohat", "ADV"), ("achi", "ADJ"), ("hai", "VB")],
    [("charger", "NN"), ("jaldi", "ADV"), ("kharab", "ADJ"), ("ho", "VB"), ("gaya", "VB")],
    [("screen", "NN"), ("toot", "VB"), ("gayi", "VB"), ("delivery", "NN"), ("mein", "PREP")],
    [("earphones", "NN"), ("ki", "PREP"), ("sound", "NN"), ("bekar", "ADJ"), ("hai", "VB")],
    [("jacket", "NN"), ("ka", "PREP"), ("color", "NN"), ("fade", "VB"), ("ho", "VB"), ("gaya", "VB")],
    # Add to ROMAN_URDU_CORPUS in corpus.py
    [("delivery", "NN"), ("bohat", "ADV"), ("late", "ADJ"), ("thi", "VB")],
    [("packing", "NN"), ("achi", "ADJ"), ("nahi", "ADV"), ("thi", "VB")],
    [("material", "NN"), ("bohat", "ADV"), ("sasta", "ADJ"), ("tha", "VB")],
    [("color", "NN"), ("fade", "VB"), ("ho", "VB"), ("gaya", "VB")],
    [("battery", "NN"), ("jaldi", "ADV"), ("khatam", "ADJ"), ("hoti", "VB")],
    [("sound", "NN"), ("bohat", "ADV"), ("achi", "ADJ"), ("hai", "VB")],
    [("charger", "NN"), ("kharab", "ADJ"), ("ho", "VB"), ("gaya", "VB")],
    [("speaker", "NN"), ("ki", "PREP"), ("quality", "NN"), ("achi", "ADJ"), ("hai", "VB")],
]
# src/corpus.py (Add this below your existing code)

# ─────────────────────────────────────────────────────────────────────────────
# CRF NER BIO-LABELLED TRAINING DATA
# Labels: O, B-BRAND, B-PLATFORM, B-LOC, B-PRODUCT, B-ASPECT
# ─────────────────────────────────────────────────────────────────────────────

BIO_TRAINING_DATA = [
    (["samsung", "ka", "phone", "acha", "hai"],
     ["B-BRAND", "O", "B-PRODUCT", "O", "O"]),
    
    (["daraz", "se", "order", "kiya", "tha", "lekin", "delivery", "late", "thi"],
     ["B-PLATFORM", "O", "O", "O", "O", "O", "B-ASPECT", "O", "O"]),
    
    (["screen", "ki", "quality", "zabardast", "hai"],
     ["B-PRODUCT", "O", "B-ASPECT", "O", "O"]),
    
    (["karachi", "mein", "tcs", "ki", "service", "achi", "hai"],
     ["B-LOC", "O", "B-PLATFORM", "O", "B-ASPECT", "O", "O"]),
    
    (["infinix", "ki", "battery", "bohat", "jaldi", "khatam", "hoti", "hai"],
     ["B-BRAND", "O", "B-PRODUCT", "O", "O", "O", "O", "O"]),
    
    (["laptop", "ki", "qeemat", "kaafi", "zyada", "hai", "magar", "design", "pyara", "hai"],
     ["B-PRODUCT", "O", "B-ASPECT", "O", "O", "O", "O", "B-ASPECT", "O", "O"]),
    
    (["apple", "watch", "ka", "strap", "kharab", "nikla"],
     ["B-BRAND", "B-PRODUCT", "O", "B-PRODUCT", "O", "O"]),
    
    (["lahore", "mein", "parcel", "2", "din", "mein", "mila", "packing", "achi", "thi"],
     ["B-LOC", "O", "O", "O", "O", "O", "O", "B-ASPECT", "O", "O"]),
    
    (["shirt", "ka", "rang", "utar", "gaya", "aur", "stitching", "bhi", "kharab", "hai"],
     ["B-PRODUCT", "O", "B-ASPECT", "O", "O", "O", "B-PRODUCT", "O", "O", "O"]),
    
    (["mobile", "ka", "kamera", "aur", "sound", "behtareen", "hai"],
     ["B-PRODUCT", "O", "B-PRODUCT", "O", "B-ASPECT", "O", "O"])
]