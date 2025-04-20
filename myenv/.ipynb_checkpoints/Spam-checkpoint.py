import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset (adjust the file path as needed)
df = pd.read_csv('spam.csv', encoding='latin-1')

# Check the first few rows to understand the structure
print(df.head())

# Clean the text data (remove special characters, digits, etc.)
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Tokenize the cleaned text and remove stopwords
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['cleaned_text'].apply(tokenize_and_remove_stopwords)

# Split the data into training and testing sets (80/20 split)
X = df['tokens']
y = df['label']  # Assuming 'label' column is 'spam' or 'ham'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the cleaned data
print(X_train.head())
