
from flask import Flask, request, jsonify
import spacy_udpipe
# import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS
load_dotenv()
import os

CORS(app) 


app = Flask(__name__)

SECRET_TOKEN = os.getenv("API_KEY")
def require_token(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token.split(' ')[1]  # Extract the token part after 'Bearer '
            if token == SECRET_TOKEN:
                return func(*args, **kwargs)
        return jsonify({'message': 'Authentication failed'}), 401

    return decorated

# @app.route('/api/resource', methods=['GET'])




def clean_text(input_text):
    # Define regex patterns for email, phone, URL, and website
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))'
    url_pattern = r"\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s!()[]{};:'\".,<>?«»“”‘’]))"
    website_pattern = r"(http|https)://[\w-]+(.[\w-]+)+\S*"

    cleaned_text = re.sub(email_pattern, '', input_text)

    cleaned_text = re.sub(phone_pattern, '', cleaned_text)

    cleaned_text = re.sub(url_pattern, '', cleaned_text)

    cleaned_text = re.sub(website_pattern, '', cleaned_text)

    return cleaned_text



 
@app.route('/predict', methods=['POST'])
@require_token
def analyze_text_combined():
    text = request.json['text']
    cleaned_text = clean_text(text)


    lang_tokenizer = AutoTokenizer.from_pretrained("mwz/LanguageDetection")
    lang_model = AutoModelForSequenceClassification.from_pretrained("mwz/LanguageDetection")

    inputs = lang_tokenizer(cleaned_text, return_tensors="pt")
    outputs = lang_model(**inputs)
    predicted_language_idx = outputs.logits.argmax().item()
    languages = ["English", "Roman Urdu", "Urdu"]
    detected_lang = languages[predicted_language_idx]


    # Sentiment Analysis
    sentiment_tokenizer = None
    sentiment_model = None

    if detected_lang == "Urdu":
        sentiment_tokenizer = AutoTokenizer.from_pretrained("mwz/UrduClassification")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("mwz/UrduClassification")
    elif detected_lang == "Roman Urdu":
        sentiment_tokenizer = AutoTokenizer.from_pretrained("mwz/RomanUrduClassification")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("mwz/RomanUrduClassification")
    else:
        sentiment_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")

    sentiment_input = sentiment_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    sentiment_logits = sentiment_model(**sentiment_input).logits
    sentiment_pred = sentiment_logits.argmax().item()

    if detected_lang == "Urdu":
        sentiment_label = "Positive" if sentiment_pred == 0 else "Negative"
    elif detected_lang == "Roman Urdu":
        sentiment_label = "Negative" if sentiment_pred == 0 else ("Neutral" if sentiment_pred == 1 else "Positive")
    else:
        sentiment_label = "Negative" if sentiment_pred == 0 else "Positive"

    # Verb Classification
    spacy_models = {
        "English": spacy_udpipe.load("en"),
        "Roman Urdu": spacy_udpipe.load("ur"),
        "Urdu": spacy_udpipe.load("ur"),
    }

    nlp = spacy_models.get(detected_lang)
    if nlp is None:
        print("Language not supported")
        return

    doc = nlp(text)

    verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            verbs.append(token.text)


    # Actionable Words for Roman Urdu
    actionable_file_path = "actionable.txt"  # Update this with the correct path
    with open(actionable_file_path, "r",encoding="utf8") as actionable_file:
      actionable_words_roman = set(line.strip() for line in actionable_file)

    actionable_words_ru = [word for word in actionable_words_roman if word in text.lower()]

    actionable_words = []
    if detected_lang == "Roman Urdu":
        actionable_words = actionable_words_ru
    else:
      actionable_words = verbs

    # Profanity Classification
    profanity_file_path = "All_Profane.txt"  # Update this with the correct path
    with open(profanity_file_path, "r",encoding="utf8") as profanity_file:
        profanity_words = [line.strip() for line in profanity_file]

    masked_text = cleaned_text
    masked_words = []

    for word in profanity_words:
        regex = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        masked_word = get_replacement_for_swear_word(word)

        if regex.search(masked_text):
            masked_text = regex.sub(masked_word, masked_text)
            masked_words.append(word)

    return {
        "Text": cleaned_text,
        "Detected Language": detected_lang,
        "Sentiment": sentiment_label,
        "Actionable words": actionable_words,
        "Profanities": masked_words,
    }

def get_replacement_for_swear_word(word):
    return word[:1] + "*" * (len(word) - 2) + word[-1:]

# Example usage
if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')
    
