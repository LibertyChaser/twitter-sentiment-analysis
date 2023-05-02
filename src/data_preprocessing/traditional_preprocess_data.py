import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

def preprocess_data(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path for the CSV file in the data folder
    csv_file_path = os.path.join(script_dir, '..', '..', 'data', 'processed', file_name)

    # Load the cleaned Sentiment140 dataset
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    
    # Remove rows with NaN values
    df.dropna(inplace=True)

    # Reduce the dataset to 1/20 of its original size
    df = df.sample(frac=0.05, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Create a CountVectorizer object and fit it to the training data
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    # Create a TfidfVectorizer object and fit it to the training data
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_cv, X_train_tfidf, y_train, X_test_cv, X_test_tfidf, y_test
