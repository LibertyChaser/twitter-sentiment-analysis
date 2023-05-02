import html
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from mapping import EMOTICON_MAPPING

tqdm.pandas()

# Initialize stop words and lemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def read_csv(file_name, folder='raw'):
    # Get the current script's absolute directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path for the CSV file in the data folder
    csv_file_path = os.path.join(script_dir, '..', '..', 'data', folder, file_name)

    columns = ['label', 'tweet_id', 'date', 'query', 'user', 'text']

    # Read the CSV file using pandas with the detected encoding
    return pd.read_csv(csv_file_path, encoding='ISO-8859-1', header=None, names=columns)

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)

    # Handle emoticons
    for emoticon, replacement in EMOTICON_MAPPING.items():
        tweet = re.sub(emoticon, replacement, tweet)

    # Handle abbreviations (this can be extended with more abbreviations)
    abbreviation_mapping = {
        "u": "you",
        "r": "are",
        "y": "why",
        "cuz": "because",
        "thx": "thanks",
        "pls": "please",
        "plz": "please",
        "w/": "with",
        "btw": "by the way",
        "b4": "before",
        "gr8": "great",
        "l8r": "later",
        "ttyl": "talk to you later"
    }
    for abbreviation, replacement in abbreviation_mapping.items():
        tweet = re.sub(r'\b{}\b'.format(abbreviation), replacement, tweet, flags=re.IGNORECASE)
        
    # Remove HTML encoding
    tweet = html.unescape(tweet)

    # Tokenize the tweet
    words = word_tokenize(tweet)

    # Remove URLs, mentions, and hashtags
    words = [re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\S+|@\w+|#\w+', '', word) for word in words]

    # Remove punctuation and special characters
    words = [re.sub(r'\W', ' ', word) for word in words]

    # Remove stopwords
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into a cleaned tweet
    cleaned_tweet = ' '.join(words)
    
    # Remove multiple spaces
    cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet).strip()

    return cleaned_tweet


def process_and_save_sentiment140():
    print("Processing raw Sentiment140 dataset...")
    sentiment140_df = read_csv("testdata.manual.2009.06.14.csv")
    
    # Clean the tweets with a progress bar
    sentiment140_df['text'] = sentiment140_df['text'].apply(clean_tweet)
    
    # Save the cleaned dataset to the data/processed/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, '..', '..', 'data', 'processed', 'cleaned_sentiment140.csv')
    sentiment140_df.to_csv(output_file_path, index=False)
    print("Finished processing and saved cleaned dataset.")

def get_sentiment140():
    try:
        # Attempt to read the cleaned dataset from the data/processed/ folder
        sentiment140_df = read_csv("cleaned_sentiment140.csv", folder='processed')
    except FileNotFoundError:
        # If the cleaned dataset is not available, process the raw dataset
        process_and_save_sentiment140()
        
        # Read the cleaned dataset from the data/processed/ folder
        sentiment140_df = read_csv("cleaned_sentiment140.csv", folder='processed')
    
    return sentiment140_df


if __name__ == "__main__":
    sentiment140_df = get_sentiment140()
    print(sentiment140_df.head())
