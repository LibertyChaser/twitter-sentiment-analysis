import nltk
from data_preprocessing.sentiment_vectors_preprocess import preprocess_data
from model_training_evaluation.training_models import train_models

nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')

def main():
    # Preprocess the data
    X_train_cv, X_train_tfidf, y_train, X_test_cv, X_test_tfidf, y_test = preprocess_data('cleaned_training_sentiment140.csv')
    # X_train_cv, X_train_tfidf, y_train, X_test_cv, X_test_tfidf, y_test = preprocess_data('cleaned_sentiment140.csv')

    # Train and evaluate models using CountVectorizer
    print("\nTraining and evaluating models using CountVectorizer:")
    train_models(X_train_cv, y_train, X_test_cv, y_test)

    # Train and evaluate models using TfidfVectorizer
    print("\nTraining and evaluating models using TfidfVectorizer:")
    train_models(X_train_tfidf, y_train, X_test_tfidf, y_test)


if __name__ == '__main__':
    main()
