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
- Sentiment140: [http://help.sentiment140.com/for-students/](http://help.sentiment140.com/for-students/)
- SentiStrength: [http://sentistrength.wlv.ac.uk/documentation/TwitterDataset.html](http://sentistrength.wlv.ac.uk/documentation/TwitterDataset.html)

2. Place the downloaded datasets in the `data` folder and run the preprocessing scripts for each dataset:

```bash
python preprocess_sentiment140.py
python preprocess_senti_strength.py
```

This will generate cleaned CSV files for each dataset.

## Model Training and Evaluation

1. Extract emotional features from the cleaned datasets:

```bash
python extract_features.py
```

2. Train and evaluate the sentiment classification models:

```bash
python train_and_evaluate.py
```

## Contributing

We welcome contributions to this project. If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them with a descriptive commit message.
4. Push your changes to your forked repository.
5. Create a pull request on the original repository.

We will review your pull request and merge your changes if everything looks good. Please ensure your code follows best practices and is properly documented.
