A Machine Learning-Based Multiclass Classification Framework for Lateral Myocardial Infarction Using Clinical and ECG Data

Abstract:

This project explores the development of a predictive machine learning model to classify the type of lateral myocardial infarction (MI) using a structured clinical and electrocardiographic (ECG) dataset. The dataset comprises detailed patient records, including demographic, historical, ECG, and biochemical features. Using exploratory data analysis (EDA), feature selection, and supervised learning models (SVM and Random Forest), we developed an end-to-end pipeline to handle imbalanced data, optimize model performance, and interpret key predictive features.

1. Introduction:

Cardiovascular diseases, particularly myocardial infarctions, are among the leading causes of mortality globally. ECG and clinical history remain key tools for diagnosis. This study aims to apply machine learning to classify the type of lateral MI based on historical patient data. Our focus is on building a robust and interpretable model pipeline capable of handling class imbalance, optimizing accuracy, and delivering clinically meaningful insights.

2. Dataset Overview:

* **Source:** `Ass.csv` (with definitions from `codebookAss2.pdf`)
* **Records:** Each row represents a unique patient encounter
* **Features:** 100+ features including demographics, cardiac history, ECG patterns, lab values
* **Target Variable:** `lat_im` (indicating the type of lateral MI: values 0–4)

3. Methodology:

3.1 Data Preprocessing:

* **ID column removed** as it holds no predictive value.
* **Missing values** handled based on skewness:

  * Median imputation for skewed numerical columns
  * Mean imputation for normally distributed features
  * Mode imputation for categorical features

3.2 Exploratory Data Analysis (EDA):

* Distributions visualized via histograms and boxplots
* Correlation heatmaps constructed to detect multicollinearity
* Highly correlated features (threshold > 0.6) were dropped

3.3 Handling Class Imbalance:

* Since the target variable (`lat_im`) was imbalanced, we applied:

  * `class_weight='balanced'` in both SVM and Random Forest
  * This ensures higher penalties on misclassified minority classes during training

3.4 Feature Scaling:

* All numeric features were scaled using `StandardScaler` for better performance with SVM

4. Model Training and Evaluation:

4.1 Support Vector Machine (SVM):

* GridSearchCV used to tune `C`, `gamma`, and `kernel`
* Best parameters selected based on 5-fold cross-validation
* Evaluation metrics:

  * Accuracy
  * Confusion Matrix
  * Classification Report
  * F1-scores (macro, micro, weighted)

4.2 Random Forest Classifier:

* GridSearchCV applied to optimize `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`
* Feature importances visualized to interpret top predictors
* Same evaluation metrics as SVM used

4.3 Multiclass Performance Visualization:

* Per-class precision, recall, and F1-scores plotted
* Confusion matrices saved as heatmaps for both models

5. Results:

🔍 SVM Results:
Best Params: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy: 0.6264705882352941
F1-score (macro): 0.29850991114149006
F1-score (micro): 0.6264705882352941
F1-score (weighted): 0.6047380483293796
Classification Report:
               precision    recall  f1-score   support

         0.0       0.65      0.59      0.62       115
         1.0       0.67      0.78      0.72       184
         2.0       0.09      0.05      0.07        19
         3.0       0.14      0.07      0.09        15
         4.0       0.00      0.00      0.00         7

    accuracy                           0.63       340
   macro avg       0.31      0.30      0.30       340
weighted avg       0.59      0.63      0.60       340


🌳 Random Forest Results:
Best Params: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Accuracy: 0.7441176470588236
F1-score (macro): 0.3162773029439696
F1-score (micro): 0.7441176470588236
F1-score (weighted): 0.6962314954471817
Classification Report:
               precision    recall  f1-score   support

         0.0       0.77      0.80      0.79       115
         1.0       0.73      0.88      0.80       184
         2.0       0.00      0.00      0.00        19
         3.0       0.00      0.00      0.00        15
         4.0       0.00      0.00      0.00         7

    accuracy                           0.74       340
   macro avg       0.30      0.34      0.32       340
weighted avg       0.66      0.74      0.70       340


✅ Processed dataset saved as processed_data.csv

* **Random Forest** showed slightly better overall accuracy and interpretability via feature importances
* **SVM** performed competitively and benefited from hyperparameter tuning and scaling

6. Visual Outputs:

* `target_distribution.png` – Class balance
* `svm_confusion_matrix.png` / `rf_confusion_matrix.png` – Confusion matrices
* `svm_metrics_per_class.png` / `rf_metrics_per_class.png` – Precision, recall, F1-score per class
* `feature_importances_rf.png` – Top predictors in Random Forest
* `processed_data.csv` – Cleaned dataset

7. Conclusion:

This study successfully demonstrated how machine learning can be applied to complex clinical datasets to support diagnosis and decision-making in cardiology. The classification of lateral myocardial infarction using SVM and Random Forest models showed promising results, with Random Forest yielding more interpretable insights.

Future improvements may include:

* Using SMOTE or ADASYN for synthetic oversampling
* Adding ROC-AUC curve analysis for each class
* Testing deep learning models for improved performance
