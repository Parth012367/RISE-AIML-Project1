
import joblib
import re

model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

message = input("Enter your message: ")
cleaned_message = clean_text(message)

vector = vectorizer.transform([cleaned_message])

prediction = model.predict(vector)

if prediction[0] == 1:
    print("Prediction: SPAM 🚫")
else:
    print("Prediction: HAM ✅")
