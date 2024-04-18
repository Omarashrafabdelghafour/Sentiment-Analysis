import pandas as pd
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [word for word in tokens if word not in string.punctuation]

    return ' '.join(tokens)

# Load your data here if it's not already loaded
data = pd.read_csv("Restaurant_Reviews.csv")

# Preprocess the text data
data['Review'] = data['Review'].apply(preprocess_text)

# Instantiate and fit CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['Review'])

# target column
y = data['Liked']
# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Train the data using RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Create a function to make predictions
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    numerical_record = vectorizer.transform([preprocessed_text])
    prediction = classifier.predict(numerical_record)
    return prediction

# Streamlit app code
import streamlit as st

def main():
    st.title("Sentiment Analysis")
    user_input = st.text_input("Enter a sentence:")
    if st.button("Predict"):
        prediction = predict_sentiment(user_input)
        st.write("Predicted label:", prediction)

if __name__ == "__main__":
    main()