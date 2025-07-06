from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess('data/spam.csv')

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

joblib.dump(model, 'models/spam_classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
