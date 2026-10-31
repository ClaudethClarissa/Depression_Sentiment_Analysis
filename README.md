# Detecting Depression-Related Language in Tweets Using Machine Learning

A machine-learning exploration of **depression-related language in tweets** using decision trees and n-gram features (unigrams, bigrams, and trigrams).

## Research question

Can the language used in a tweet help predict its dataset label (depression-related versus non-depression-related)?

**Scope:** This is **tweet-level text classification**, not clinical diagnosis or a tool for determining whether an individual has depression. Ground-truth labels describe the dataset's annotation process; they are not verified clinical diagnoses.

## Dataset

The repository includes `sentiment_tweets3.csv` with `message` (tweet text) and `label` (binary class) columns. The analysis checks class balance and exact duplicate messages, and removes duplicated text before splitting the data to reduce train/test overlap. Labels and sampling criteria should be assessed before interpreting results beyond the supplied dataset.

## Methods

The notebook compares three configurations of n-gram features (1, 2 and 3 words), selecting the five most frequent n-grams **within each class from training data only** (up to ten binary indicators). The same selected features are then applied to the holdout test set. A decision tree predicts the binary label, and results include accuracy, balanced accuracy, class-specific precision/recall/F1 and confusion matrices.

The data split is stratified, with 80% used for training and 20% for testing. Any repeated tweet text is handled before splitting to avoid placing copies in both partitions. Feature selection never uses holdout labels.

## Ablation experiment: removing explicit depression terms

A separate experiment removes the tokens **depression**, **depressed**, and **depressive** from both training and test tweets, before extracting the n-grams. It then repeats the unigram experiment on the same split.

This tests the model's reliance on explicit depression-related vocabulary. A decrease in accuracy would suggest these words are highly informative for the classification task; it would **not**, by itself, indicate that the original model was invalid. Self-references to depression can be meaningful language evidence for tweet classification.

The notebook reports the measured metrics when executed. Earlier accuracy figures from the original notebook should **not** be treated as validated performance estimates because its feature selection separately used test-set labels.

## How to run

1. Install Python 3.10+ and the packages `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `jupyter`.
2. Place `sentiment_tweets3.csv` beside `Sentiment_Analysis_Tw.ipynb` (both are included here).
3. Open and run the notebook from top to bottom.

## Interpretation and limitations

- The classifier predicts **dataset labels**, not a diagnosis or an individual's mental health status.
- Tweets mentioning depression may be legitimately informative, but general discussion of depression does not necessarily convey a personal experience.
- Class imbalance, annotation and sampling methodology, near-duplicates, and shifts to new data may affect generalization.
- Comparing performance with and without explicit terms is a **sensitivity analysis**, not a substitute for external validation.
- An independent, carefully annotated test set would be needed before making stronger generalization claims.

## Academic context

This project was completed as coursework for **Fundamentals of Network Analysis and Data Mining**, taught by **Professor Emma Towlson** at the **University of Calgary**. The present repository includes a methodological review and revised evaluation workflow.
