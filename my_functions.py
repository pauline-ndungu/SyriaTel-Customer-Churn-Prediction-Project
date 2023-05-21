import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
