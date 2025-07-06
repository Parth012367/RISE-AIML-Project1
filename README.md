# 📧 Email Spam Detection using Machine Learning

This project builds a machine learning model to automatically classify emails as **spam** or **ham (not spam)** using **TF-IDF vectorization** and the **Naive Bayes classifier**. It also includes a script to test your own messages.

---

## 🧠 Objective

Spam emails impact productivity and pose a security risk. This project aims to:
- Detect spam automatically using machine learning
- Train a lightweight model with over 90% accuracy
- Provide a script to predict messages without retraining

---

## 📁 Project Structure

```
Project 1/
│
├── data/
│ └── spam.csv # Dataset file (ham/spam)
│
├── models/
│ ├── spam_classifier.pkl # Trained model (saved)
│ └── vectorizer.pkl # TF-IDF vectorizer (saved)
│
├── preprocess.py # Text cleaning & preprocessing
├── train_model.py # Train, evaluate, and save model
├── predict.py # Predict for new input messages
└── requirements.txt # Python dependencies
```

---

## 📦 Installation

1. **Clone this project or copy the folder**
2. Open terminal inside the folder
3. Set up virtual environment:

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# OR
source venv/bin/activate    # On Mac/Linux
```
4. Install dependencies:

```bash
pip install -r requirements.txt
```
5. Train the model:

```bash
python train_model.py
```
6. Test with your own messages:

```bash
python predict.py
```