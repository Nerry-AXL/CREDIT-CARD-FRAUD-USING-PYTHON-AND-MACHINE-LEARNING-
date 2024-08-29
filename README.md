# Credit Card Fraud Detection

This repository contains a project focused on detecting credit card fraud using Python and machine learning techniques. The aim is to build a model that can accurately distinguish between fraudulent and non-fraudulent transactions, thereby aiding in the prevention of financial fraud.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Credit card fraud is a significant problem for financial institutions and consumers alike. This project utilizes machine learning algorithms to detect fraudulent transactions. The project includes data preprocessing, model training, and evaluation of different machine learning models to determine which one performs best in identifying fraudulent transactions.

## Dataset

The dataset used for this project is publicly available and contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

- **Number of transactions:** 284,807
- **Number of fraudulent transactions:** 492
- **Features:** 30 features, including `Time`, `Amount`, and 28 anonymized features labeled `V1`, `V2`, ..., `V28`.

The dataset can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle.

## Installation

To run this project, you'll need to install the necessary Python libraries. You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

To use this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Nerry-AXL/CREDIT-CARD-FRAUD-USING-PYTHON-AND-MACHINE-LEARNING-.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd CREDIT-CARD-FRAUD-USING-PYTHON-AND-MACHINE-LEARNING-
    ```

3. **Open the Jupyter notebook:**

    ```bash
    jupyter notebook CREDIT_CARD_FRAUD_DETECTION.ipynb
    ```

4. **Run the notebook:**
   Execute the cells in the notebook to preprocess the data, train the models, and evaluate their performance.

## Model Building

The project explores various machine learning models, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)
- Neural Networks

Feature engineering and selection are also performed to improve model performance.

## Evaluation

The models are evaluated using several metrics, including:

- **Accuracy:** The ratio of correctly predicted transactions to the total transactions.
- **Precision:** The ratio of correctly predicted fraudulent transactions to all predicted fraudulent transactions.
- **Recall:** The ratio of correctly predicted fraudulent transactions to all actual fraudulent transactions.
- **F1 Score:** The weighted average of Precision and Recall.
- **ROC-AUC:** The Area Under the Receiver Operating Characteristic curve.

Due to the imbalanced nature of the dataset, emphasis is placed on precision, recall, and F1 score.

## Results

The final model achieves high accuracy in detecting fraudulent transactions with minimal false positives. The best-performing model is selected based on the evaluation metrics mentioned above.
