import os
import pandas as pd
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def preprocess_data(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path for the CSV file in the data folder
    csv_file_path = os.path.join(script_dir, '..', '..', 'data', 'processed', file_name)

    # Load the cleaned Sentiment140 dataset
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    
    # Remove rows with NaN values
    df.dropna(inplace=True)

    # Reduce the dataset to 1/10 of its original size
    df = df.sample(frac=0.05, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # POS tagging and lemmatization
    lemmatizer = WordNetLemmatizer()
    
    # Function to convert NLTK POS tags to WordNet POS tags
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    # Function to process a single tweet and return its sentiment feature vector
    def process_tweet(tweet):
        # Perform POS tagging
        tagged_words = pos_tag(tweet.split())

        # Initialize sentiment vector and POS counts
        sentiment_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pos_count = [0, 0, 0, 0]

        # Iterate over words and their POS tags
        for word, tag in tagged_words:
            # Convert the POS tag to WordNet format
            wn_pos = get_wordnet_pos(tag)

            # If the POS tag is valid, proceed with lemmatization and sentiment scoring
            if wn_pos is not None:
                # Perform lemmatization using the Morphy lookup method
                lemma = lemmatizer.lemmatize(word, pos=wn_pos)

                # Retrieve sentiment scores for the lemma and POS from SentiWordNet
                senti_synsets = list(swn.senti_synsets(lemma, pos=wn_pos))

                # If sentiment scores are available, update sentiment vector and POS counts
                if senti_synsets:
                    pos_score = sum([ss.pos_score() for ss in senti_synsets]) / len(senti_synsets)
                    neg_score = sum([ss.neg_score() for ss in senti_synsets]) / len(senti_synsets)

                    if wn_pos == wn.NOUN:
                        pos_count[0] += 1
                    elif wn_pos == wn.ADJ:
                        pos_count[1] += 1
                    elif wn_pos == wn.VERB:
                        pos_count[2] += 1
                    elif wn_pos == wn.ADV:
                        pos_count[3] += 1

                    # Update sentiment vector with neutral, positive, and negative sentiment scores
                    sentiment_vector[0] += (1 - pos_score - neg_score)  # neutral_score
                    sentiment_vector[1] += pos_score
                    sentiment_vector[2] += neg_score

        # Calculate POS word frequency ratios and update sentiment vector
        total_words = sum(pos_count)
        if total_words > 0:
            sentiment_vector[3:] = [count / total_words for count in pos_count]

        # Initialize CountVectorizer and TfidfVectorizer objects
        count_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfVectorizer()

        # Fit and transform the text data using the CountVectorizer and TfidfVectorizer objects
        X_train_cv = count_vectorizer.fit_transform(X_train)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_cv = count_vectorizer.transform(X_test)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Return the processed data
        return X_train_cv, X_train_tfidf, y_train, X_test_cv, X_test_tfidf, y_test


    # Apply the process_tweet function to the training and testing sets
    X_train = X_train.apply(process_tweet)
    X_test = X_test.apply(process_tweet)

    # Initialize CountVectorizer and TfidfVectorizer objects
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the text data using the CountVectorizer and TfidfVectorizer objects
    X_train_cv = count_vectorizer.fit_transform(X_train)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_cv = count_vectorizer.transform(X_test)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Return the processed data
    return X_train_cv, X_train_tfidf, y_train, X_test_cv, X_test_tfidf, y_test
