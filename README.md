# Twitter Sentiment Analysis

This project aims to explore sentiment classification methods based on emotion feature vectors for tweets. We use various datasets, such as Sentiment140, SentiStrength, and others, to train and evaluate our models.

## Table of Contents

- [Installation](#installation)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Contributing](#contributing)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/LibertyChaser/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate # For Unix-based systems
venv\Scripts\activate # For Windows
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Collection and Preprocessing

1. Download the datasets:
- [Sentiment140](http://help.sentiment140.com/for-students/)

2. Place the downloaded datasets in the `data` folder and run the preprocessing scripts for each dataset:

```bash
python get_sentiment140.py
```

This will generate cleaned CSV files for each dataset.

## Model Training and Evaluation

### Models

The models are evaluated using cross-validation and various metrics such as accuracy, precision, recall, and F1-score. The best-performing model is chosen based on its performance on the testing set. The following models are trained and evaluated in this project:

- Multinomial Naive Bayes
- k-Nearest Neighbors
- Logistic Regression
- Support Vector Machine (linear kernel)
- Support Vector Machine (RBF kernel)
- k-Means clustering

### Evaluation

#### Traditional Model

CountVectorizer: 

| Model                           | Average Score | Accuracy | Precision (0) | Recall (0) | F1-score (0) | Precision (4) | Recall (4) | F1-score (4) |
| ------------------------------- | ------------- | -------- | ------------- | ---------- | ------------ | ------------- | ---------- | ------------ |
| Multinomial Naive Bayes         | 0.7458        | 0.75     | 0.72          | 0.79       | 0.75         | 0.77          | 0.71       | 0.74         |
| k-Nearest Neighbors             | 0.6473        | 0.65     | 0.63          | 0.69       | 0.66         | 0.67          | 0.60       | 0.63         |
| Logistic Regression             | 0.7535        | 0.75     | 0.76          | 0.73       | 0.74         | 0.74          | 0.78       | 0.76         |
| Support Vector Machine (linear) | 0.7510        | 0.75     | 0.78          | 0.70       | 0.74         | 0.73          | 0.80       | 0.77         |
| Support Vector Machine (RBF)    | 0.7161        | 0.72     | 0.71          | 0.73       | 0.72         | 0.73          | 0.71       | 0.72         |

TfidfVectorizer:

| Model                           | Average Score | Accuracy | Precision (0) | Recall (0) | F1-score (0) | Precision (4) | Recall (4) | F1-score (4) |
| ------------------------------- | ------------- | -------- | ------------- | ---------- | ------------ | ------------- | ---------- | ------------ |
| Multinomial Naive Bayes         | 0.7410        | 0.74     | 0.71          | 0.80       | 0.75         | 0.77          | 0.69       | 0.73         |
| k-Nearest Neighbors             | 0.6456        | 0.65     | 0.61          | 0.81       | 0.69         | 0.72          | 0.49       | 0.58         |
| Logistic Regression             | 0.7553        | 0.76     | 0.76          | 0.73       | 0.75         | 0.75          | 0.78       | 0.76         |
| Support Vector Machine (linear) | 0.7402        | 0.74     | 0.78          | 0.68       | 0.73         | 0.72          | 0.81       | 0.76         |
| Support Vector Machine (RBF)    | 0.7284        | 0.73     | 0.72          | 0.73       | 0.73         | 0.74          | 0.72       | 0.73         |

#### Sentimental Vector

CountVectorizer: 



TfidfVectorizer:



## Contributing

Welcome contributions to this project. If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them with a descriptive commit message.
4. Push your changes to your forked repository.
5. Create a pull request on the original repository.

I will review your pull request and merge your changes if everything looks good. Please ensure your code follows best practices and is properly documented.
