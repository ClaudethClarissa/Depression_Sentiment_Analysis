# Depression-Related Tweet Classification | NLP & Machine Learning

**Python · Natural Language Processing · N-grams · Decision Trees · Classification**

## Overview

This project explores whether textual features extracted from tweets can distinguish two labeled classes: **0 (non-depression)** and **1 (depression-related)** in the supplied dataset. It compares unigram, bigram, and trigram representations using a decision-tree classifier.

The work originated as a course assignment in **Network Analysis and Data Mining at the University of Calgary**, taught by **Professor Emma Towlson**. The notebook documents an exploratory machine-learning exercise, **not** a clinically validated depression detection system.

[**Read the original Jupyter notebook**](Sentiment_Analysis_Tw.ipynb)

## Research Questions

- Can frequently occurring n-grams be used as predictive features for the dataset's tweet labels?
- How do decision-tree predictions change when features are based on one-, two-, or three-word sequences?
- What do confusion matrices reveal beyond overall classification accuracy?
- What methodological safeguards are needed before these results can be interpreted as generalizable?

## Dataset

The repository includes [`sentiment_tweets3.csv`](sentiment_tweets3.csv), which the notebook reads with pandas. The visible columns are:

| Column | Role |
|---|---|
| `message` | Tweet text |
| `label` | Binary target: 0 or 1 |
| `Index` | An identifier/index field in the CSV |

The original dataset's provenance, collection procedure, labeling criteria, and licensing have not yet been independently verified. **These labels must not be interpreted as clinical diagnoses of the individuals who wrote the tweets.** The source file may contain user-generated text; review redistribution permissions and privacy before sharing or extending the dataset.

## Methods

### 1. Train–test split

The notebook uses `sklearn.model_selection.train_test_split` with **80% training** and **20% testing** (`random_state=42`).

### 2. N-gram extraction and feature construction

A custom function tokenizes text by spaces, removes NLTK English stopwords, and extracts contiguous n-grams. The analysis considers:

- **Unigrams (1-grams):** individual retained tokens
- **Bigrams (2-grams):** adjacent token pairs
- **Trigrams (3-grams):** adjacent token triples

For each representation, the code selects the five most frequent sequences from each labeled class and creates **10 binary text-presence features** (five for label 1 and five for label 0). A feature indicates whether a selected sequence appears in the tweet.

**Implementation detail:** the original function consistently constructs the neutral-class vocabulary using unigrams, even during the bigram and trigram comparisons. This means those experiments do not use perfectly symmetric n-gram representations across both classes.

### 3. Model and evaluation

A `DecisionTreeClassifier(random_state=42)` is fitted to the binary features. The notebook reports **accuracy, a classification report, confusion matrices, and decision-tree visualizations** for each representation.

## Results Recorded in the Notebook

These figures are taken from the notebook's **saved outputs**. They have not been independently re-executed and, importantly, **they are not valid held-out performance estimates** for the reason explained below.

| Feature setting | Saved accuracy |
|---|---:|
| Unigrams | 96.90% |
| Bigrams | 36.69% |
| Trigrams | 78.09% |

The saved confusion matrices (rows = true label, columns = predicted label; class order 0, 1) are:

| N-grams | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| 1 | 1,614 | 0 | 64 | 385 |
| 2 | 481 | 1,133 | 173 | 276 |
| 3 | 1,607 | 7 | 445 | 4 |

**Interpreting the outputs:** the unigram result appears strongest by accuracy, but the apparent result cannot be used as evidence of a reliable classifier. The trigram confusion matrix, for example, shows that only **4 of 449** positive-class examples were identified in the saved run.

### Critical evaluation limitation: target leakage

The feature-construction function `from_tweets2features(X, Y, n)` selects the most frequent class-specific n-grams using both the provided messages **and their labels**. The notebook calls it *separately on training data and on test data*, passing `Yraw_test` into the test feature construction. Consequently, the test representation is influenced by the actual test labels. The features also represent different selected vocabularies in the training and test sets.

**Therefore, the reported test accuracy and confusion matrices cannot be treated as unbiased generalization estimates.** The code is preserved as the original course analysis, and the recorded values are presented for historical inspection only, not as validated model performance.

A sound follow-up would learn the vocabulary **exclusively from the training split**, apply exactly that fixed vocabulary to validation/test messages without their labels, and re-evaluate using stratified splits, cross-validation and appropriate class-sensitive metrics (precision, recall, F1 and balanced accuracy).

## Notebook & Project Structure

```text
Depression_Sentiment_Analysis/
├── Sentiment_Analysis_Tw.ipynb  # Original feature engineering, model and results
├── sentiment_tweets3.csv       # Dataset used by the notebook
└── README.md                   # Project description and methodological notes
```

The notebook uses **pandas, NumPy, matplotlib, scikit-learn and NLTK**. To explore the original code, install those packages, download the NLTK English stopwords corpus if needed, and open the notebook from the repository root. Because the original notebook has not been freshly executed in a clean environment, package compatibility is not guaranteed.

## Ethical and Statistical Limitations

- Language patterns in tweets cannot establish whether a person has depression; clinical diagnosis requires appropriate professional assessment.
- The label definition and representativeness of the dataset are not documented here.
- The training/test feature-generation process leaks label information and invalidates the apparent test performance.
- The top-feature selection and substring-presence implementation can be sensitive to tokenization choices.
- Accuracy alone is especially misleading when classes are imbalanced; positive-class recall and precision matter.
- The published CSV contains social-media text; any downstream reuse should respect privacy and dataset terms.

## Academic Context & Acknowledgment

Developed as an assignment for **Network Analysis and Data Mining**, **University of Calgary**, under **Professor Emma Towlson**. The n-gram function in the notebook references an [Analytics Vidhya tutorial on n-grams](https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/).

---

**Portfolio:** [Data Science & Analytics Portfolio](https://github.com/ClaudethClarissa/data-science-portfolio)
