```python
# Sentiment Analysis Pipeline for Women Safety Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset loading
def run_sentiment_pipeline():
    data = {
        'text': [
            "Police patrolling increased near metro stations in Delhi, feeling much safer now.",
            "Street lights are not working in dark lanes, dangerous for night commuters.",
            "Public transport options need better security cameras installed.",
            "Great initiative by local authorities for women safety awareness campaigns."
        ],
        'sentiment': [1, 0, 0, 1]  # 1: Positive/Safe, 0: Negative/Unsafe
    }

    df = pd.DataFrame(data)

    # Feature Extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    # Model Training
    model = LogisticRegression()
    model.fit(X, y)

    # Predictions
    preds = model.predict(X)
    print("Model Accuracy:", accuracy_score(y, preds))

if __name__ == "__main__":
    run_sentiment_pipeline()
