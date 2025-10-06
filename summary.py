import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# # # Download NLTK resources (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

# # Load both datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")
#
# # Add labels
fake_df["label"] = 1  # 1 = Fake
true_df["label"] = 0  # 0 = True
#
# # Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
#
# # Show summary
print("\n🔍 Dataset Info:")
print(df.info())
# print("\n🧱 Sample Rows:")
print(df.head())
print(df.isnull().sum())
#
#
# # Set of stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

import re

# Preprocessing function without nltk.word_tokenize
def clean_tokens(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)  # removes all non-alphanumeric characters except space

    # Split into tokens using .split()
    tokens = text.split()

    # Remove stopwords
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    return cleaned_tokens

df['title_tokens'] = df['title'].apply(clean_tokens)
df['text_tokens'] = df['text'].apply(clean_tokens)

# Preview cleaned output
print("\n✅ Cleaned Data Sample:")
print(df[['title', 'title_tokens', 'text_tokens', 'label']].head())

df['clean_text'] = df['title_tokens'].apply(lambda x: " ".join(x)) + " " + df['text_tokens'].apply(lambda x: " ".join(x))

import numpy as np

# Load GloVe
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings("glove.6B/glove.6B.100d.txt")

def text_to_glove_vector(text, embeddings, dim=100):
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

# Apply to all cleaned text
df['glove_vector'] = df['clean_text'].apply(lambda x: text_to_glove_vector(x, glove_embeddings))


X = np.stack(df['glove_vector'].values)
y = df['label'].values


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['True', 'Fake'])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

import seaborn as sns
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='YlGnBu')
plt.title("Classification Report Heatmap")
plt.savefig("classification_report_heatmap.png")
plt.show()


label_counts = df['label'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(label_counts, labels=['Fake', 'True'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title("Label Distribution (Fake vs True)")
plt.savefig("label_distribution_pie.png")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("PCA of Text Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("text_embedding_pca.png")
plt.show()
