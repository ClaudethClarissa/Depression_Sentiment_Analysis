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

## Corrected Evaluation

The original notebook contained target leakage: it selected feature vocabularies independently on the test set using test labels. That issue has now been corrected in the main notebook:

1. Split tweets 80/20 using the original fixed random seed.
2. Learn five frequent n-grams from **each class in training data only**.
3. Apply the **same ten training-defined features** to training and test tweets, without test labels.
4. Use the same n-gram order in **both classes** for unigrams, bigrams, and trigrams.
5. Fit three decision trees and calculate test accuracy, confusion matrices, precision, recall, and F1.

The corrected notebook deliberately clears the previous output cells. **The original accuracies (96.90%, 36.69%, 78.09%) are invalid because of leakage and are not presented as current performance.** A GitHub Actions workflow executes the corrected notebook and stores an executed notebook artifact for inspection.

### Results from the executed, corrected notebook

The automated run completed successfully with the original 80/20 split (2,063 test examples: 1,614 in class 0 and 449 in class 1).

| Representation | Test accuracy | Class 1 precision | Class 1 recall | Class 1 F1 |
|---|---:|---:|---:|---:|
| Unigrams | 99.47% | 100% | 97.55% | 98.76% |
| Bigrams | 84.49% | 100% | 28.73% | 44.64% |
| Trigrams | 82.40% | 100% | 19.15% | 32.09% |

Confusion matrices (rows = actual labels, columns = predictions; class 0 then class 1):

| Representation | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Unigrams | 1,614 | 0 | 11 | 438 |
| Bigrams | 1,614 | 0 | 320 | 129 |
| Trigrams | 1,614 | 0 | 363 | 86 |

Results are from a [successful GitHub Actions notebook execution](https://github.com/ClaudethClarissa/Depression_Sentiment_Analysis/actions/runs/41234705361), **not** the invalid outputs of the original notebook. The unusually high unigram score is explained in large part by the word `depression`, which occurs in nearly all positive-class tweets and no negative-class tweets; see the ablation below. The run identified **10 identical text values shared between training and test**; these may contribute to optimistic estimates and should be addressed in a future group-aware evaluation.

### Dataset audit: why the original accuracy was unusually high

Inspecting the supplied 10,313-row dataset exposed a powerful **label-word shortcut**:

| Dataset property | Label 0 | Label 1 |
|---|---:|---:|
| Tweets | 8,000 | 2,313 |
| Tweets containing the word `depression` | 0 | 2,287 |
| Proportion mentioning `depression` | 0.00% | 98.88% |

As a result, a model can largely predict the **dataset's label** by detecting the literal word *depression*. This does not demonstrate that it recognizes depression in people or generalizable linguistic signs of depression.

The notebook now contains an **ablation experiment** masking that exact word in both training and test text and repeating all three models:

| Experiment | Accuracy | Label-1 recall | Label-1 TP / FN |
|---|---:|---:|---:|
| Unigrams, original text | 99.47% | 97.55% | 438 / 11 |
| Unigrams, `depression` masked | **88.37%** | **47.44%** | 213 / 236 |
| Bigrams, `depression` masked | 83.42% | 24.05% | 108 / 341 |
| Trigrams, `depression` masked | 82.40% | 19.15% | 86 / 363 |
| Masked unigrams, duplicate-grouped 80/20 split | **87.87%** | **44.19%** | 194 / 245 |

The final experiment groups exact normalized tweet texts before splitting so that duplicate texts cannot appear in both partitions. The original split had 10 identical text values shared across training and test; the dataset overall also contains duplicate records. In the duplicate-grouped evaluation, the test set comprised 2,061 rows instead of 2,063.

**Interpretation:** removing a single label-indicating word causes a large reduction in positive-class sensitivity. Even the masked results should be viewed as exploratory, since other labeling or sampling artifacts may remain. More rigorous dataset provenance assessment, held-out-source validation, and repeated group-aware splits would be needed before making performance or mental-health claims.

The masked and grouped results were verified in the [automated notebook execution](https://github.com/ClaudethClarissa/Depression_Sentiment_Analysis/actions/runs/41236198512).

The experiment still uses a simple frequency-selected feature representation and a single 80/20 holdout split. These are exploratory results, not clinical validation. The notebook also reports identical tweet texts shared across train and test, which may warrant a group-aware split if duplicates occur.

## Notebook & Project Structure

```text
Depression_Sentiment_Analysis/
├── Sentiment_Analysis_Tw.ipynb  # Original feature engineering, model and results
├── sentiment_tweets3.csv       # Dataset used by the notebook
├── .github/workflows/validate.yml # Automated notebook execution
└── README.md                   # Project description and methodological notes
```

The notebook uses **pandas, NumPy, matplotlib and scikit-learn**; the corrected version uses scikit-learn's built-in English stopword list instead of requiring a downloaded NLTK corpus. To explore the original code, install those packages, open the notebook from the repository root. The GitHub Actions workflow validates notebook execution in a clean Python environment.

## Ethical and Statistical Limitations

- Language patterns in tweets cannot establish whether a person has depression; clinical diagnosis requires appropriate professional assessment.
- The label definition and representativeness of the dataset are not documented here.
- The earlier test-label leakage was fixed by learning vocabulary from training data only; old saved accuracy values must not be reused.
- The simple tokenization, stopword filtering, and top-five feature selection can affect predictive performance.
- Accuracy alone is especially misleading when classes are imbalanced; positive-class recall and precision matter.
- The published CSV contains social-media text; any downstream reuse should respect privacy and dataset terms.

## Academic Context & Acknowledgment

Developed as an assignment for **Network Analysis and Data Mining**, **University of Calgary**, under **Professor Emma Towlson**. The original n-gram function referenced an [Analytics Vidhya tutorial on n-grams](https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/).

---

**Portfolio:** [Data Science & Analytics Portfolio](https://github.com/ClaudethClarissa/data-science-portfolio)
