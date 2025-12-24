# Titanic Survival Classification: A Multi-Model Comparison

This repository contains a comprehensive Jupyter Notebook that implements and compares five different machine learning classification algorithms to predict passenger survival on the Titanic. The project highlights the importance of data preprocessing and feature scaling in model performance.

## 🚀 Project Overview

The goal is to determine which classification algorithm best handles the Titanic dataset features (Age, Sex, Class, Fare, etc.) to predict survival. This notebook serves as a complete pipeline from raw data cleaning to final model evaluation.

## 📊 Dataset

The project uses the **Titanic dataset** (via Seaborn).
- **Initial Samples:** 891
- **Cleaned Samples:** 782 (after duplicate removal and handling missing values)
- **Features:** Passenger Class, Sex, Age, SibSp, Parch, Fare, and Embarked.

## 🛠️ Data Pipeline & Preprocessing

* **Data Cleaning:** * Removed duplicates and redundant features (`deck`, `who`, `adult_male`, `class`, `alive`, `embark_town`).
    * Imputed missing `Age` values using the mean.
    * Handled missing `Embarked` values by dropping null rows.
* **Encoding:** Used `LabelEncoder` for categorical strings (`Sex`, `Embarked`).
* **Feature Scaling:** Applied `StandardScaler` to ensure features like `Fare` and `Age` are on a comparable scale—this is critical for the distance-based algorithms like KNN and SVM used in this project.

## 🤖 Models Implemented

The following models were trained and evaluated:

1.  **Logistic Regression:** Used as the baseline statistical model.
2.  **K-Nearest Neighbors (KNN):** Evaluated with various values of $k$ (e.g., $k=5$).
3.  **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' Theorem.
4.  **Support Vector Machine (SVM):** A powerful classifier used with an optimized kernel to find the best decision boundary.
5.  **Decision Tree Classifier:** Included for hierarchical decision-making analysis.

## 📈 Performance & Results

Models were compared using the following metrics:
* **Accuracy:** The percentage of correct predictions.
* **Precision, Recall, & F1-Score:** To measure the model's balance between finding all survivors and being correct about those it labels as survivors.
* **Classification Reports:** Detailed performance per class (Survived vs. Not Survived).

**Key Performance Highlight:** The models achieved accuracy scores ranging from **75% to over 80%**, with the **SVM** and **KNN** models benefiting significantly from the feature scaling steps.

## 💻 Technical Stack

- **Python**
- **Pandas & NumPy:** Data manipulation
- **Matplotlib & Seaborn:** Data visualization
- **Scikit-Learn:** Preprocessing (`StandardScaler`, `LabelEncoder`) and Machine Learning models.

## 📖 How to Run

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/yourusername/titanic-survival-benchmark.git](https://github.com/yourusername/titanic-survival-benchmark.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib
    ```
3.  **Run the Notebook:**
    Launch `classification_models.ipynb` in your preferred environment.


## 🫡 New Update 

- **Added Cross Validation too**
- **To access that file use ->** classification_models_withCrossValidation.ipynb
