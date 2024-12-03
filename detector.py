import os
import time
import string
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pyrogram import Client, filters

# Download stopwords
nltk.download('stopwords')

# Preprocess dataset
df = pd.read_csv("https://raw.githubusercontent.com/silverfruitplayer/email-detector/main/spam_ham_dataset.csv")
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

corpus = []
for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def get_new_emails(service, label_ids=['INBOX']):
    results = service.users().messages().list(userId='me', labelIds=label_ids).execute()
    messages = results.get('messages', [])
    return messages

# Function to classify email
def classify_new_email(email_text):
    email_text = email_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    email_text = ' '.join(email_text)

    email_corpus = [email_text]
    X_email = vectorizer.transform(email_corpus)
    predicted_label = mnb.predict(X_email)[0]
    return "Spam" if predicted_label == 1 else "Ham"

# Telegram Bot setup
api_id = '6'
api_hash = 'eb06d4abfb49dc3eeb1aeb98ae0f581e'
bot_token = ''

app = Client("spamdetectbot", api_id=api_id, api_hash=api_hash, bot_token=bot_token)

@app.on_message(filters.command("start"))
async def start(client, message):
    await message.reply("Welcome! The bot is now monitoring your Gmail inbox for new emails.")

    service = authenticate_gmail()
    processed_messages = set()

    while True:
        try:
            messages = get_new_emails(service)
            for msg in messages:
                msg_id = msg['id']
                if msg_id not in processed_messages:
                    email = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                    snippet = email['snippet']
                    processed_messages.add(msg_id)

                    result = classify_new_email(snippet)
                    await message.reply(f"New Email:\n\nSnippet: {snippet}\nClassification: {result}")
            time.sleep(30)
        except Exception as e:
            await message.reply(f"Error: {str(e)}")
            time.sleep(60)

@app.on_message(filters.command("classify") & filters.text)
async def classify_handler(client, message):
    email_text = message.text.replace("/classify", "").strip()
    if email_text:
        result = classify_new_email(email_text)
        await message.reply(f"Classification: {result}")
    else:
        await message.reply("Please provide email text after the /classify command.")

app.run()
