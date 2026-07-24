# Starbucks Capstone Challenge

Machine learning project for predicting customer response to Starbucks Rewards offers using simulated mobile app event data.

Blog post: [Predicting Starbucks Offer Response with Machine Learning](https://medium.com/@renan.cuoghi/predicting-starbucks-offer-response-with-machine-learning-f9d62641a639)

## Project Overview

Starbucks periodically sends mobile app users different offer types, including BOGO, discount, and informational offers. Not every customer receives the same offer, and not every customer responds.

This project builds a supervised machine learning workflow to predict whether a customer will successfully respond to a received offer. It also analyzes which demographic groups respond best to each offer type.

## Problem Statement

For each `offer received` event, the goal is to predict whether the customer will respond successfully during the offer validity window.

A successful response is defined as:

- **BOGO and discount offers:** the customer views the offer and completes that same offer before it expires.
- **Informational offers:** the customer views the offer and then makes a transaction before it expires.

The final dataset has one row per offer exposure and uses only information available before the offer is sent.

## Repository Structure

```text
.
├── Starbucks_Capstone_notebook.ipynb   # Main analysis and modeling notebook
├── README.md                           # Project documentation
├── data/
│   ├── portfolio.json                  # Offer metadata
│   ├── profile.json                    # Customer demographics
│   └── transcript.json                 # Customer event records
```

## Data

The project uses three JSON datasets:

| File | Description |
|---|---|
| `portfolio.json` | Metadata for 10 offers, including offer type, reward, difficulty, duration, and channels. |
| `profile.json` | Demographic data for 17,000 customers, including age, gender, income, and membership date. |
| `transcript.json` | 306,534 records for transactions, offers received, offers viewed, and offers completed. |

Records with missing demographic data were removed from the modeling dataset. These records are represented by `age = 118`, missing `gender`, and missing `income`.

## Methodology

The notebook includes:

- Data loading and inspection.
- Cleaning customer demographics.
- Normalizing nested transcript event values.
- Engineering offer, customer, membership, and channel features.
- Building response labels from offer validity windows.
- Exploratory visualizations.
- Benchmark modeling with a majority-class classifier.
- Logistic regression baseline.
- Random forest model and hyperparameter tuning.
- Final evaluation and segment analysis.

## Models and Metrics

The primary metric is **F1 score**, because the model needs to balance identifying likely responders with avoiding unnecessary offer sends. Accuracy and ROC-AUC are also reported.

Final held-out test results:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority benchmark | 0.577 | 0.000 | 0.500 |
| Logistic regression | 0.645 | 0.603 | 0.705 |
| Initial random forest | 0.695 | 0.650 | 0.757 |
| Tuned random forest | 0.690 | 0.657 | 0.755 |

The selected model is the tuned random forest.

Best tuned parameters:

```text
max_depth: 8
min_samples_leaf: 15
min_samples_split: 10
```

## Key Findings

- The tuned random forest substantially outperformed the majority-class benchmark.
- Membership tenure, social channel delivery, membership year, income, offer duration, difficulty, and reward were among the most important predictors.
- BOGO and discount offers had strong response rates among women aged 31-45 in higher income groups.
- Informational offers showed lower spend after view, which is expected because they do not provide a direct reward.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start Jupyter:

```bash
jupyter notebook
```

Open and run:

```text
Starbucks_Capstone_notebook.ipynb
```

The notebook is already executed and includes saved outputs.

## Limitations

- The dataset is simulated and simplified.
- The model predicts response, not true incremental lift.
- Some customers may have purchased even without receiving an offer.
- The objective optimizes response, not profit, margin, or long-term customer value.
- Missing demographic records were removed, which may introduce bias.

## Future Improvements

- Use uplift modeling with randomized holdout groups to estimate causal impact.
- Optimize for expected profit instead of response probability.
- Add customer transaction history features before each offer.
- Compare additional algorithms such as gradient boosting.
- Track model performance separately by offer type and demographic segment.
