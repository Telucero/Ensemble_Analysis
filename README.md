# Ensemble Heart Analysis
Overview
This repository contains scripts for building and evaluating machine learning models to predict a binary outcome variable (DEATH_EVENT) using a dataset on heart failure clinical records. The scripts employ various ensemble learning techniques, including bagging, boosting, and stacking, to enhance the predictive power of the models.

# Baseline Script
The baseline script performs the following steps:
Load the dataset using pandas from a CSV file.
Rename variables for clarity.
Apply standardization to selected columns using StandardScaler from scikit-learn.
Split the data into training, validation, and test sets.
Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution in the training set.
Model Building:
Train three types of models: Simple Neural Network (SimpleNN), XGBoost, and Support Vector Machine (SVM).
Evaluate each model on the validation set using metrics such as accuracy, precision, and recall.
Evaluate each model on the test set to assess its generalization performance.
Generate confusion matrices for each model on the test set to visualize classification results.

# Bagging:
Use the BaggingClassifier from scikit-learn to create ensemble models for each base model (SimpleNN, XGBoost, SVM).
Train ensemble models for each base model by creating an ensemble of five classifiers.
Evaluate the performance of each Bagging ensemble model on both the validation and test sets.
Compute metrics such as accuracy, precision, and recall for each Bagging ensemble model.
Generate confusion matrices for each Bagging ensemble model on the test set to visually represent classification results.
## Impact and Results:
Bagging did not significantly alter the performance of the original models. While it may lead to performance improvements in some cases, for this specific dataset and evaluation metrics, the original models were already performing optimally, and the addition of Bagging did not yield further enhancements.

# Boosting 
The boosting script enhances the original models using the AdaBoost boosting technique. The boosting script includes the following steps:

## AdaBoost:
Apply AdaBoost to each base model (SimpleNN, XGBoost, SVM) to boost their performance.
Evaluate the performance of each boosted model on both the validation and test sets.
Compute metrics such as accuracy, precision, and recall for each boosted model.
Generate confusion matrices for each boosted model on the test set to visually represent classification results.
## Impact and Results:
AdaBoost had a varied impact on the performance of the original models:
For SimpleNN, AdaBoost slightly decreased the recall while maintaining accuracy and precision.
For XGBoost, AdaBoost led to a decrease in accuracy, precision, and recall.
For SVM, AdaBoost resulted in a decrease in accuracy, precision, and recall.


# Stacking:
Train diverse base models (SimpleNN, XGBoost, SVM) on the full dataset.
Combine predictions of base models using logistic regression as the final estimator.
Evaluate the performance of the stacking aggregated model on the test set.
Compute metrics such as accuracy, precision, and recall for the stacking aggregated model.
Compare the performance of the stacking aggregated model to the original models.
## Impact and Results:
Stacking produced comparable performance to the original models:
Stacking achieved similar accuracy and precision as the original XGBoost and SVM models.
Stacking maintained the recall score consistent with the original models.
## Conclusion
The ensemble heart analysis scripts demonstrate the application of various ensemble learning techniques to improve the predictive power of machine learning models for heart failure prediction. Each technique had a different impact on the performance of the original models, with stacking showing the most promising results. Below are the numerical evaluation values for each model:

