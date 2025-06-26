
import gradio as gr
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

nltk.download("vader_lexicon")

#  Initialize VADER
sia = SentimentIntensityAnalyzer()

#  Load RoBERTa
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
roberta_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load Dataset from CSV

reviews_df = pd.read_csv("Reviews.csv").dropna(subset=["Text", "Score"])
reviews_df = reviews_df[reviews_df["Score"].isin([1, 2, 4, 5])].sample(2000, random_state=42)

# Binary sentiment: 1,2 = negative (0), 4,5 = positive (1)

reviews_df["label"] = reviews_df["Score"].apply(lambda x: 0 if x in [1, 2] else 1)
texts = reviews_df["Text"].tolist()
labels = reviews_df["label"].tolist()

# Preprocessing & Training TF-IDF + ML models
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Prediction Functions

def predict_vader(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return {"positive": score, "negative": 1 - score}
    elif score <= -0.05:
        return {"negative": -score, "positive": 1 + score}
    else:
        return {"neutral": 1.0}

def predict_roberta(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = roberta_model(**inputs)
    probs = softmax(outputs.logits.detach().numpy()[0])
    return {"negative": float(probs[0]), "neutral": float(probs[1]), "positive": float(probs[2])}

def predict_logistic(text):
    vec = vectorizer.transform([text])
    proba = logistic_model.predict_proba(vec)[0]
    return {"negative": float(proba[0]), "positive": float(proba[1])}

def predict_nb(text):
    vec = vectorizer.transform([text])
    proba = nb_model.predict_proba(vec)[0]
    return {"negative": float(proba[0]), "positive": float(proba[1])}

def get_label_and_confidence(scores):
    label = max(scores, key=scores.get)
    return label.capitalize(), round(float(scores[label]), 3)

#  Gradio Inference Function
def analyze_sentiment(text, model):
    if model == "VADER":
        scores = predict_vader(text)
    elif model == "RoBERTa":
        scores = predict_roberta(text)
    elif model == "Logistic Regression":
        scores = predict_logistic(text)
    else:
        scores = predict_nb(text)

    label, confidence = get_label_and_confidence(scores)

    # Plot with correct colors
    color_map = {"negative": "red", "neutral": "gray", "positive": "green"}
    colors = [color_map.get(label, "blue") for label in scores.keys()]
    plt.figure(figsize=(5, 3))
    plt.bar(scores.keys(), scores.values(), color=colors)
    plt.ylim(0, 1)
    plt.xlabel("Sentiment")
    plt.ylabel("Confidence")
    plt.title("Sentiment Scores")
    return label, confidence, plt

#  Gradio UI
demo = gr.Blocks()
with demo:
    gr.Markdown("# 🧐 Sentiment Analysis App (Logistic Regression, Naive bayes, Vader, Roberta)")
    with gr.Row():
        model_choice = gr.Dropdown(["VADER", "RoBERTa", "Logistic Regression", "Naive Bayes"], label="Select Model")
    with gr.Row():
        text_input = gr.Textbox(lines=3, placeholder="Enter your text here", label="Input Text")
    with gr.Row():
        label_output = gr.Textbox(label="Predicted Sentiment")
        confidence_output = gr.Textbox(label="Confidence")
    chart_output = gr.Plot()
    analyze_button = gr.Button("Analyze")

    analyze_button.click(analyze_sentiment, inputs=[text_input, model_choice], outputs=[label_output, confidence_output, chart_output])

demo.launch(share = True)
