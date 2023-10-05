

<h1 align="center">SyriaTel Customer Churn Project</h1>


## :dart: Project Title and Description ##


This project aims to predict customer churn in a telecom company using machine learning techniques.

The dataset used for this project is the "Syrian Telecom Churn Dataset," which contains information about customers and their interaction with telecommunication services.

## Project Structure

The project is organized into the following files and directories:

- README.md: Provides an overview of the project, its purpose, and instructions on how to use it.

- telecom_churn.ipynb: Jupyter Notebook containing the code, analysis, and modeling steps.


## :rocket: Technologies ##

The following tools were used in this project:

- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.

## :white_check_mark: Requirements ##

- Clone the project repository to your local machine or download the project files.

- Install the required Python libraries if you haven't already.
-  You can use the following command:

   $ pip install pandas numpy matplotlib seaborn scikit-learn


## :checkered_flag: Getting Started ##


- Clone this project 
- git clone <Repository url>
   
- Open the telecom_churn.ipynb notebook using Jupyter Notebook.

- Run the code cells in the notebook sequentially to execute the analysis and modeling steps.

- Review the results, including visualizations and performance metrics of different models.

- Modify the code or experiment with different models and techniques as needed.

## Model Evaluation

Several machine learning models were trained and evaluated using the dataset to predict customer churn.

The following models were implemented and their performance metrics were measured:

* Logistic Regression

* Decision Tree

* Random Forest

* Gradient Boosting

The evaluation metrics used to assess the models include:

* Accuracy: The proportion of correctly predicted churned and non-churned customers.

* Precision: The proportion of correctly predicted churned customers out of all predicted churned customers.

* Recall: The proportion of correctly predicted churned customers out of all actual churned customers.

* F1-score: The harmonic mean of precision and recall, providing a balanced measure between the two.

- To ensure the robustness of the models' performance, cross-validation was performed. The dataset was divided into multiple folds, and each model was trained and evaluated multiple times, with different folds used for training and testing. This helps to assess the generalization capability of the models.

- The model with the best performance can be identified based on the chosen evaluation metric and the mean score obtained from cross-validation.

Please refer to the code and results in the telecom_churn.ipynb notebook for detailed information on each model's performance and cross-validation scores
   

![modelevaluation](https://github.com/pauline-ndungu/Project/assets/125973188/e54de8ea-ecd6-4014-9cfa-8cb11fbdef7d)
![Feature Importance](https://github.com/pauline-ndungu/Project/assets/125973188/54e381fd-61f9-4bf5-8027-47570b637073)

![Bimodel Evaluation](https://github.com/pauline-ndungu/Project/assets/125973188/5d0ae8b3-a53b-458b-bb10-326214df3034)

## Conclusion

In conclusion, this project demonstrated the process of predicting customer churn in a telecom company using machine learning techniques. By analyzing the Telecom Churn Dataset and training different models, it was possible to assess the performance of each model using evaluation metrics such as accuracy, precision, recall, and F1-score. Additionally, cross-validation was performed to validate the models' performance and ensure their generalization capability.



&#xa0;

<a href="#top">Back to top</a>

