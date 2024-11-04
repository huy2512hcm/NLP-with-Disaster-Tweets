# Natural Language Processing with Disaster Tweets

**Kaggle Competition:** [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

## Project Overview

This project tackles the challenge of classifying tweets as disaster-related or not, using various Natural Language Processing (NLP) techniques. By experimenting with different feature extraction methods and machine learning models, we aim to achieve high accuracy and robust generalization in predicting disaster-related content.

### Dataset

- The dataset includes a collection of tweets, each labeled as either related to a disaster (1) or unrelated (0).
- **Training Set Size:** Approximately 7,613 tweets
- **Test Set Size:** Approximately 3,263 tweets
- **Goal:** Develop a classifier to predict whether each tweet pertains to a real disaster.

## Methodology

### 1. Data Preprocessing
The tweet data is preprocessed to enhance model performance:
- **Lowercasing**: All text converted to lowercase for uniformity.
- **Tokenization and Lemmatization**: Text is tokenized, and words are lemmatized to reduce inflectional forms.
- **Noise Removal**: Removed URLs, mentions, stopwords, and punctuation to eliminate noise and improve signal clarity.

### 2. Feature Extraction
Two main feature extraction methods were used:
- **Bag-of-Words (BOW)**: Converted tweets into binary word presence vectors with a minimum frequency threshold to filter rare words.
- **N-grams**: Extended feature vectors with bigrams to capture short word sequences, enhancing contextual representation.

### 3. Model Training and Evaluation
Three classification models were implemented and evaluated:
- **Logistic Regression (L1 & L2 Regularization)**: Trained on both BOW and N-gram features to minimize overfitting while capturing relevant patterns.
- **Bernoulli Naive Bayes**: Implemented with Laplace smoothing to handle zero-frequency issues, aiming to balance recall and precision.

Each model’s performance was assessed using the F1-score, favoring models that balance precision and recall due to the importance of accurate disaster prediction.

### 4. Model Performance
The best-performing model achieved an F1 score of 0.79098 on the public leaderboard. The combination of regularization and N-gram features proved effective in handling noisy Twitter data and providing robust predictions.

## Technologies Used
- **Python Libraries**: NumPy, Scikit-learn, NLTK, Pandas
- **Machine Learning Models**: Logistic Regression, Bernoulli Naive Bayes
- **NLP Techniques**: Bag-of-Words, N-grams

## Results

The final model demonstrated effective generalization and an impressive public leaderboard score of **0.79098** on Kaggle, suggesting strong predictive capabilities on real-world disaster data.

## Future Improvements
To further improve accuracy, consider:
- **Deep Learning Models**: Exploring advanced models like LSTM or transformers.
- **Data Augmentation**: Increasing dataset diversity to improve model robustness.