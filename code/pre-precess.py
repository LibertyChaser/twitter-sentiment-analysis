import tweepy
import csv
import pandas as pd
import re
from credentials import consumer_key, consumer_secret, access_token, access_secret

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def collect_tweets_from_ids(ids, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tweet_id', 'text', 'label'])

        for tweet_id in ids:
            try:
                tweet = api.get_status(tweet_id, tweet_mode='extended')
                text = clean_tweet(tweet.full_text)
                # Set the label based on the dataset (0: negative, 2: neutral, 4: positive)
                label = '0'
                writer.writerow([tweet_id, text, label])
            except tweepy.TweepError as e:
                print(f"Failed to retrieve tweet ID {tweet_id}: {e}")


def main():
    # Load tweet IDs from Sentiment140 and OMD datasets
    sentiment140_ids = pd.read_csv('../data/cleaned_sentiment140_ids.csv')['tweet_id'].tolist()
    omd_ids = pd.read_csv('omd_ids.csv')['tweet_id'].tolist()

    # Collect and pre-process tweets
    collect_tweets_from_ids(sentiment140_ids, 'sentiment140_tweets.csv')
    collect_tweets_from_ids(omd_ids, 'omd_tweets.csv')


if __name__ == '__main__':
    main()
