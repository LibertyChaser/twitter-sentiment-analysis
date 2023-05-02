import pandas as pd
import re

def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def main():
    # Load and preprocess the Sentiment140 dataset
    columns = ['label', 'tweet_id', 'date', 'query', 'user', 'text']
    data = pd.read_csv('../data/originaldata/trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None, names=columns)
    
    # Clean the tweet text
    data['text'] = data['text'].apply(clean_tweet)
    
    # Save cleaned dataset to a new CSV file
    data.to_csv('../data/cleaned_sentiment140.csv', index=False)

if __name__ == '__main__':
    main()
