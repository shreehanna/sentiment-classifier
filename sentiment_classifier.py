from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# === TRAINING DATA ============================================

# Input text data (you can later load )
texts = [
    "I love pizza",
    "This movie is great",
    "I hate homework",
    "This is so boring"
]

# Labels: 1 = positive, 0 = negative
labels = [1, 1, 0, 0]

# === TEXT VECTORIZATION =======================================

# Convert text into numerical feature vectors
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)

# === MODEL TRAINING ===========================================

# Initialize and train the classifier
model = LogisticRegression()
model.fit(X_train, labels)

# === TESTING / INFERENCE ======================================

# New text inputs to analyze
new_texts = [
    "I love The Devil Wears Prada",
    "This is Trash"
]

# Convert new text to same vector space as training data
X_test = vectorizer.transform(new_texts)

# Predict sentiment
predictions = model.predict(X_test)

# === OUTPUT RESULTS ===========================================

# Interpret and print predictions
for sentence, label in zip(new_texts, predictions):
    if label == 1:
        mood = "😄 Positive"
    else :
        mood= "😡 Negative"
    print(f"{mood} → \"{sentence}\"")




