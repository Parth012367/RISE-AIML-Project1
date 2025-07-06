import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_and_preprocess(path):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.read_csv(path, encoding='latin-1', usecols=[0, 1], names=['label', 'text'], header=None, skiprows=1)

    df.dropna(subset=['text', 'label'], inplace=True)

    df['text'] = df['text'].apply(clean_text)

    print("\nSample cleaned texts:")
    print(df[['label', 'text']].head(10))

    df = df[df['text'].str.strip().astype(bool)]

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, vectorizer
