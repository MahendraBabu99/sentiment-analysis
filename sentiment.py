# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# 📌 Step 2: Load All 3 CSVs
filenames = [
    "archive/1429_1.csv",
    "archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    "archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
]

dfs = []
for file in filenames:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# 📌 Step 3: Combine DataFrames
df = pd.concat(dfs, ignore_index=True)

# 📌 Step 4: Keep Only Required Columns
for col in df.columns:
    if 'review' in col.lower() and 'text' in col.lower():
        review_col = col
    if 'rating' in col.lower():
        rating_col = col

df = df[[review_col, rating_col]].dropna()
df.columns = ['review', 'rating']

# 📌 Step 5: Convert Ratings to Sentiment
def get_sentiment(rating):
    try:
        rating = float(rating)
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
    except:
        return None

df['sentiment'] = df['rating'].apply(get_sentiment)
df = df.dropna(subset=['sentiment'])

# 📌 Step 6: Text Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_review'] = df['review'].apply(preprocess)

# 📌 Step 7: Balance the Classes
df_positive = df[df.sentiment == 'positive']
df_negative = df[df.sentiment == 'negative']
df_neutral = df[df.sentiment == 'neutral']

min_len = min(len(df_positive), len(df_negative), len(df_neutral))

df_balanced = pd.concat([
    resample(df_positive, replace=False, n_samples=min_len, random_state=42),
    resample(df_negative, replace=False, n_samples=min_len, random_state=42),
    resample(df_neutral, replace=False, n_samples=min_len, random_state=42)
])

df_balanced = df_balanced.sample(frac=1, random_state=42)

# 📌 Step 8: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_balanced['cleaned_review']).toarray()
y = df_balanced['sentiment']

# 📌 Step 9: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 10: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 📌 Step 11: Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("confusion_matrix.png")
plt.show()
