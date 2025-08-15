# Credit Scoring Project

This project provides a comprehensive machine learning pipeline for credit scoring, predicting loan default risk.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)




## Project Structure

```
credit_scoring_project/
├── credit_risk_dataset.csv         # Raw dataset
├── data_summary.txt                # Summary of data exploration
├── evaluate_models.py              # Script to evaluate trained models
├── explore_data.py                 # Script for exploratory data analysis (EDA)
├── Makefile                        # Makefile for project automation
├── preprocess_data.py              # Script for data cleaning and preprocessing
├── run_all.py                      # Main script to run the entire pipeline
├── requirements.txt                # Python dependencies
├── models/                         # Directory to store trained models
│   ├── decision_tree_model.pkl
│   ├── logistic_regression_model.pkl
│   └── random_forest_model.pkl
├── processed_data/                 # Directory to store processed data
│   ├── X_test_scaled.csv
│   ├── X_train_scaled.csv
│   ├── y_test.csv
│   └── y_train.csv
└── screenshots/
    ├── 1.jpg
    └── 2.jpg
```




## Setup

To set up the project, follow these steps:

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd credit_scoring_project
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install -U pip
    pip install -r requirements.txt
    ```




## Usage

To run the entire credit scoring pipeline, execute the `run_all.py` script:

```bash
python run_all.py
```

Alternatively, you can use the `Makefile` for common operations:

-   **Install dependencies:**

    ```bash
    make install
    ```

-   **Run the full pipeline:**

    ```bash
    make run
    ```

-   **Clean generated files:**

    ```bash
    make clean
    ```




## Pipeline Overview

The `run_all.py` script orchestrates the following steps:

1.  **`preprocess_data.py`**: Handles data loading, cleaning (unrealistic ages, missing values), encoding categorical features, splitting data into training and testing sets, and scaling numerical features. The processed data is saved in the `processed_data/` directory.

2.  **`explore_data.py`**: Performs exploratory data analysis on the raw dataset. It displays basic information, descriptive statistics, checks for missing values, and saves a summary to `data_summary.txt`.

3.  **`train_models.py`**: Trains multiple machine learning models (Logistic Regression, Decision Tree, Random Forest) using the preprocessed training data. The trained models are saved as `.pkl` files in the `models/` directory.

4.  **`evaluate_models.py`**: Loads the trained models and evaluates their performance on the test set using metrics such as accuracy, precision, recall, and F1-score. It also prints a detailed classification report for each model.




## Models Used

The project trains and evaluates the following machine learning models:

-   **Logistic Regression**: A linear model for binary classification, often used as a baseline.
-   **Decision Tree Classifier**: A non-linear model that makes decisions based on a tree-like structure.
-   **Random Forest Classifier**: An ensemble method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.




## Evaluation Metrics

The models are evaluated using the following metrics:

-   **Accuracy**: The proportion of correctly classified instances.
-   **Precision**: The ratio of true positive predictions to the total predicted positives. Useful when the cost of false positives is high.
-   **Recall (Sensitivity)**: The ratio of true positive predictions to the total actual positives. Useful when the cost of false negatives is high.
-   **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
-   **Classification Report**: Provides a detailed breakdown of precision, recall, and F1-score for each class, along with support (number of actual occurrences of the class).




## Results

After running the `evaluate_models.py` script, the performance metrics for each trained model will be printed to the console. An example output might look like this:

```
📊 Model: logistic_regression_model.pkl
Accuracy : 0.8500
Precision: 0.8000
Recall   : 0.7500
F1-Score : 0.7742

Detailed Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.92      0.90      1000
           1       0.80      0.75      0.77       500

    accuracy                           0.85      1500
   macro avg       0.84      0.83      0.83      1500
weighted avg       0.85      0.85      0.85      1500
```

*(Note: The exact values will depend on the dataset and model training. The above is an illustrative example.)*




## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.




## License

This project is licensed under the MIT License. See the `LICENSE` file for details.




## Screenshots

Here are some screenshots of the project:

### Screenshot 1

![Screenshot 1](https://private-us-east-1.manuscdn.com/sessionFile/O0jP7fB5xk9a940vzgM3Ip/sandbox/Oks7jC821PUBD3aKX3DtXT-images_1755256846807_na1fn_L2hvbWUvdWJ1bnR1L2NvZGUtYWxwaGEtaW50cmVuc2hpcC9jcmVkaXRfc2NvcmluZ19wcm9qZWN0L3NjcmVlbnNob3RzLzE.jpg?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTzBqUDdmQjV4azlhOTQwdnpnTTNJcC9zYW5kYm94L09rczdqQzgyMVBVQkQzYUtYM0R0WFQtaW1hZ2VzXzE3NTUyNTY4NDY4MDdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTnZaR1V0WVd4d2FHRXRhVzUwY21WdWMyaHBjQzlqY21Wa2FYUmZjMk52Y21sdVoxOXdjbTlxWldOMEwzTmpjbVZsYm5Ob2IzUnpMekUuanBnIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=sRIjt-1CvFWOmDZmqIT~FNrwV7cBWI97RWFHMTvg-55pUH7CejuKlZDmOEpmTupwNVKAtGayr9v6I~6CC3-~jaFNfdy4o5-4mmwWEnAvN1OYWW9H58V3cElAZ9uI~VdWaFjz~Yz8uJQbmNvYDa8DpEup9~xtX4UPNTfX5mRS2hS98jgdQ5Ppb7pibD1i-y-5VMAB7s1edkXOgo5APCXnJr9tPGjEDLW3tfTMQMWfSoeA4vPudnTI-pBBxZs~LXpNTZHDnSeusLVhRiAYIzYXZ3pi5mNwIwcQiczQOTMpd3OimLnhoi00C7m95EWHADyGc~HA9eJMQg72c-cSViFhPg__)

### Screenshot 2

![Screenshot 2](https://private-us-east-1.manuscdn.com/sessionFile/O0jP7fB5xk9a940vzgM3Ip/sandbox/Oks7jC821PUBD3aKX3DtXT-images_1755256846808_na1fn_L2hvbWUvdWJ1bnR1L2NvZGUtYWxwaGEtaW50cmVuc2hpcC9jcmVkaXRfc2NvcmluZ19wcm9qZWN0L3NjcmVlbnNob3RzLzI.jpg?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTzBqUDdmQjV4azlhOTQwdnpnTTNJcC9zYW5kYm94L09rczdqQzgyMVBVQkQzYUtYM0R0WFQtaW1hZ2VzXzE3NTUyNTY4NDY4MDhfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTnZaR1V0WVd4d2FHRXRhVzUwY21WdWMyaHBjQzlqY21Wa2FYUmZjMk52Y21sdVoxOXdjbTlxWldOMEwzTmpjbVZsYm5Ob2IzUnpMekkuanBnIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=gwBWAvbVo83aayCMlh6OpXh2XR-F-3Kl3OZG8gFH1JO8OexN01jiTD6x7XAIAuROY~C-JOei7locQjx6H1VftIe4mkIAQH6DsRjqNCMYvJGBkWXyArG7v1JpM4rmteK1eFMV0MxWyh1XFnSqqQqZJqnnPcNarAvtT6~8qkLpLiyGUU8n8Qgs6zW0QrJQ4NorjEEn32W7ZZ1ewiRa~bfkqL5c7CXbbZHpl2e13dYc-ZMT51TmHpJJigZTSJV-JdiZANYn5DmhzUDwzDhH6ZiMoj3IMItVnxju~I~7XRJVgrmTHftc-mRIFAhAi3KjZ1p-1xOk3q~TUx9tw3KMkDGxyQ__)



