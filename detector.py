import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from pyrogram import Client, filters

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/silverfruitplayer/email-detector/main/spam_ham_dataset.csv")

# Clean the text column
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

# Initialize stemmer
stemmer = PorterStemmer()

# Preprocess the text data
corpus = []
stopwords_set = set(stopwords.words('english'))

for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Function to preprocess and classify a new email string
def classify_new_email(email_text):
    email_text = email_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    email_text = ' '.join(email_text)

    email_corpus = [email_text]
    X_email = vectorizer.transform(email_corpus)

    # Make prediction
    predicted_label = mnb.predict(X_email)[0]

    # Function to print classification result
    def classify_email(predicted):
        prediction = "Spam" if predicted == 1 else "Ham"
        return f"Predicted: {prediction}"

    # Print prediction result
    result = classify_email(predicted_label)
    return result

# Telegram Bot setup
api_id = '6'
api_hash = 'eb06d4abfb49dc3eeb1aeb98ae0f581e'
bot_token = ''

app = Client("spamdetectbot", api_id=api_id, api_hash=api_hash, bot_token=bot_token)


@app.on_message(filters.command("classify") & filters.text)
async def classify_handler(client, message):
    email_text = message.text.replace("/classify", "").strip()
    if email_text:
        result = classify_new_email(email_text)
        await message.reply(result)
    else:
        await message.reply("Please provide an email text after the /classify command.")

@app.on_message(filters.command("start"))
async def start(client, message):
    await message.reply("Welcome to the Spam/Ham Email Classifier Bot! Send /classify followed by the email text to classify it as Spam or Ham.")

app.run()
