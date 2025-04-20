import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None, names=['label', 'message'])

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_message'] = df['message'].apply(clean_text)

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.clf()

# Metric Bar Plot
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Performance")
plt.ylim(0, 1)
plt.savefig("performance_metrics.png")
plt.clf()

# WordClouds
spam_words = ' '.join(df[df['label'] == 'spam']['cleaned_message'])
ham_words = ' '.join(df[df['label'] == 'ham']['cleaned_message'])

spam_wc = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
ham_wc = WordCloud(width=800, height=400, background_color='white').generate(ham_words)

spam_wc.to_file("spam_wordcloud.png")
ham_wc.to_file("ham_wordcloud.png")
