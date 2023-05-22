import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Load the dataset into a pandas dataframe
df = pd.read_csv('Data/telecom_churn.csv')
def plot_categorical_churn(column_name):
    # Calculate the percentage of churned and non-churned customers for each category
    category_counts = df[column_name].value_counts()
    churn_counts = df.groupby(column_name)['churn'].sum()
    churn_perc = (churn_counts / category_counts) * 100
    nonchurn_perc = 100 - churn_perc

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the bars for churned and non-churned customers
    bar_width = 0.35
    bar_positions = range(len(category_counts))

    ax.bar(bar_positions, churn_perc, bar_width, label='Churned')
    ax.bar(bar_positions, nonchurn_perc, bar_width, bottom=churn_perc, label='Non-Churned')

    # Set the x-axis tick labels and plot title
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(category_counts.index)
    ax.set_xlabel(column_name)
    ax.set_ylabel('Percentage')
    ax.set_title(f'Percentage of Churned and Non-Churned Customers by {column_name}')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()





def evaluate_model(y_true, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    # Calculate F1-score
    f1 = f1_score(y_true, y_pred)
    print("F1-score:", f1)

    # Creating a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Non-Churned', 'Churned']  # Specify the class labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()

    plt.show()



def perform_cross_validation(X, y, model_type='decision_tree', n_estimators=100, max_depth=None, cv=5):
    """
    Perform cross-validation using the specified classifier and calculate the mean accuracy scores.

    Args:
    - X: The feature matrix.
    - y: The target variable.
    - model_type: The type of model to use ('decision_tree', 'random_forest', 'logistic_regression', or 'gradient_boosting').
    - n_estimators: The number of trees in the Random Forest or Gradient Boosting (default: 100).
    - max_depth: The maximum depth of each tree (default: None).
    - cv: The number of cross-validation folds (default: 5).

    Returns:
    - The mean cross-validated accuracy score.
    """

    if model_type == 'decision_tree':
        # Create the Decision Tree classifier
        clf = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == 'random_forest':
        # Create the Random Forest classifier
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif model_type == 'logistic_regression':
        # Create the Logistic Regression classifier
        clf = LogisticRegression()
    elif model_type == 'gradient_boosting':
        # Create the Gradient Boosting classifier
        clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
    else:
        raise ValueError("Invalid model_type. Supported types are 'decision_tree', 'random_forest', 'logistic_regression', and 'gradient_boosting'.")

    # Perform cross-validation
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    # Calculate the mean accuracy score
    mean_accuracy = scores.mean()

    return mean_accuracy




def plot_model_performance(models, cross_validation_scores, accuracy_scores, precision_scores, recall_scores, f1_scores):
    # Set the width of the bars
    bar_width = 0.15

    # Set the positions of the bars on the x-axis
    r1 = range(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # Set the figure size
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create the bar plot
    ax.bar(r1, cross_validation_scores, width=bar_width, label='Cross Validation Score')
    ax.bar(r2, accuracy_scores, width=bar_width, label='Accuracy')
    ax.bar(r3, precision_scores, width=bar_width, label='Precision')
    ax.bar(r4, recall_scores, width=bar_width, label='Recall')
    ax.bar(r5, f1_scores, width=bar_width, label='F1-score')

    # Set the x-axis labels
    ax.set_xticks([r + bar_width*2 for r in r1])
    ax.set_xticklabels(models)

    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Performance of Models in Predicting Churn')

    # Move the legend outside the plot
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Adjust the layout to avoid clipping the labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_recall_scores(models, recall_scores):
    # Set the figure size and create axes
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create the bar plot
    ax.bar(models, recall_scores, color='#F02C1C')

    # Set labels and title
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Recall', fontweight='bold', fontsize=14)
    ax.set_title('Best Performing Models', fontweight='bold', fontsize=20)

    # Set the x-axis tick positions and labels
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, fontweight='bold', fontsize=14)

     # Set y-axis tick customization
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # Customize the y-axis tick positions
    ax.set_yticklabels(['0.1', '0.2','0.3','0.4','0.5','0.6', '0.7', '0.8', '0.9'], fontweight='bold', fontsize=10)



    # Show the plot
    plt.show()
