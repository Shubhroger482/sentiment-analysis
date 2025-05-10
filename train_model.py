import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")
X = df['review']
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline and train
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Save model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as sentiment_model.pkl")
