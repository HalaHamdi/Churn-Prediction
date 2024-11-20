import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import nice_table
from IPython.display import display
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, learning_curve
import warnings

def evaluate(y_true, y_pred, title, table=False):
    '''
    Given the true labels and predicted ones, the binary classification evaluation metrics are returned.
    '''
    
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    
    # ROC AUC can only be computed if y_pred contains probabilities
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        roc_auc = None  # In case of binary classification with only one class present
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }

    if table:
        display(nice_table(metrics, title=title))

    return metrics

def cross_validation(clf, x_data, y_data, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    warnings.filterwarnings("ignore")
    
    # Add ROC AUC to the labels dictionary
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "roc_auc": "ROC AUC"  # New label for ROC AUC
    }

    # Perform cross-validation with the specified scoring metrics
    scores = cross_validate(clf, x_data, y_data, cv=cv, scoring=scoring, return_train_score=True)
    
    train = {}
    test = {}
    
    # Loop through all metrics in scoring and extract train/test scores
    for metric in scoring:
        train[f"{labels[metric]}_train"] = float(scores[f"train_{metric}"].mean())
        test[f"{labels[metric]}_test"] = float(scores[f"test_{metric}"].mean())

    # Display the results
    display(nice_table(train, title="Train"))
    display(nice_table(test, title="Test"))

    return {**train, **test}


def learning_curves(clf, x_data, y_data, N, scoring="f1", y_label="F1 Score"):
    '''
    Plot the learning curve for a given classification model using F1 score or Recall.
    
    Parameters:
    - clf: The classifier instance.
    - x_data: Feature data.
    - y_data: Target labels.
    - N: List or array of training sizes.
    - scoring: Scoring metric, can be 'f1' or 'recall'.
    - y_label: Label for the y-axis.
    '''
    train_sizes, train_scores, test_scores = learning_curve(clf, x_data, y_data, n_jobs=4, 
                                                            train_sizes=N, scoring=scoring)

    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.xlabel("Training Examples")
    plt.ylabel(y_label)
    plt.title("Learning Curve")
    
    # Calculate mean and standard deviation for training and test scores
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    # Plot training and test learning curves
    plt.plot(train_sizes, train_mean, markersize=5, label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, test_mean, markersize=5, label='Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.legend(loc="best")
    plt.show()

def log_weights_analysis(clf, x_data,top=20):
    '''
    Display feature importance of each KPI using the get_feature_importance function.
    '''
    # Get feature importance using the previously defined function
    importance_df = get_feature_importance(clf, x_data).head(top)
    
    # Prepare the data for plotting
    features = importance_df['Feature']
    importance_scores = importance_df['Importance Score']
    
    # Plotting
    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance_scores, color='skyblue')
    plt.axvline(x=0, color='grey', linewidth=0.8)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def get_feature_importance(model, X):
    """
    Retrieve feature importance from a trained model.

    Parameters:
    model: A trained model (e.g., LogisticRegression, SVC, RandomForestClassifier, etc.)
    X: The input features as a DataFrame or array-like structure

    Returns:
    importance_df: A DataFrame containing feature names and their importance scores
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    # Initialize an empty list to store importance scores
    importance_scores = None

    # Check model type and extract importance scores accordingly
    if hasattr(model, 'coef_'):
        # For Logistic Regression and Linear SVM
        importance_scores = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        # For Tree-based models and ensemble methods (Random Forest, Gradient Boosting)
        importance_scores = model.feature_importances_
    else:
        raise ValueError("Model type not supported for feature importance extraction.")

    # Create a DataFrame for better readability
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance Score': importance_scores
    })

    # Sort the DataFrame by Importance Score
    importance_df = importance_df.sort_values(by='Importance Score', ascending=False)

    return importance_df