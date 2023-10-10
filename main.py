# -*- coding: utf-8 -*
# !pip install transformers spacy_udpipe
# 
import spacy_udpipe
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import re

def analyze_text_combined(text):
    # Language Detection
    lang_tokenizer = AutoTokenizer.from_pretrained("mwz/LanguageDetection")
    lang_model = AutoModelForSequenceClassification.from_pretrained("mwz/LanguageDetection")

    inputs = lang_tokenizer(text, return_tensors="pt")
    outputs = lang_model(**inputs)
    predicted_language_idx = outputs.logits.argmax().item()
    languages = ["English", "Roman Urdu", "Urdu"]
    detected_lang = languages[predicted_language_idx]

    # NER Processing
    if detected_lang == "Urdu":
        tokenizer = AutoTokenizer.from_pretrained("mwz/UrduNER")
        model = AutoModelForTokenClassification.from_pretrained("mwz/UrduNER")
    else:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)

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

    sentiment_input = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
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
    with open(actionable_file_path, "r") as actionable_file:
        actionable_words_roman = [line.strip() for line in actionable_file]

    actionable_words = []
    if detected_lang == "Roman Urdu":
        actionable_words = actionable_words_roman

    # Profanity Classification
    profanity_file_path = "All_Profane.txt"  # Update this with the correct path
    with open(profanity_file_path, "r") as profanity_file:
        profanity_words = [line.strip() for line in profanity_file]

    masked_text = text
    masked_words = []

    for word in profanity_words:
        regex = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        masked_word = get_replacement_for_swear_word(word)

        if regex.search(masked_text):
            masked_text = regex.sub(masked_word, masked_text)
            masked_words.append({"word": word, "index": masked_text.index(masked_word)})

    return {
        "Detected Language": detected_lang,
        "NER Labels": ner_results,
        "Sentiment": sentiment_label,
        "Actionable words": verbs,
        "Censored Text": masked_text,
        "Profanities": masked_words,
    }

def get_replacement_for_swear_word(word):
    return word[:1] + "*" * (len(word) - 2) + word[-1:]

# Example usage
text = "کراچی کا پورٹ فاؤنٹین پوری قوت سے کام کرتے ہوئے 620 فٹ کی بلندی تک پہنچ جاتا ہے۔ یہ کراچی کے قریب واقع اویسٹر راکس کے قریب واقع ہے۔"
result = analyze_text_combined(text)
print(result)