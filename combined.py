import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import warnings
warnings.filterwarnings("ignore")

# Load the original dataset and duplicate it
original_data = pd.read_csv("Ass.csv")
data = original_data.copy()

# Drop ID column if exists
if 'ID' in data.columns:
    data.drop(columns='ID', inplace=True)

# Identify numerical and categorical columns
num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Handle missing values
for col in num_cols:
    if data[col].isnull().sum() > 0:
        skew = data[col].skew()
        fill_value = data[col].median() if abs(skew) > 1 else data[col].mean()
        data[col].fillna(fill_value, inplace=True)

for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Drop highly correlated numeric features
corr_matrix = data[num_cols].corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.6)]
data.drop(columns=to_drop, inplace=True)

# Define target and features
target = 'lat_im'
X = data.drop(columns=target)
y = data[target]

# Visualize class distribution
plt.figure()
sns.countplot(x=y)
plt.title("Target Distribution (lat_im)")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- SVM ----------
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svm_grid = GridSearchCV(SVC(class_weight='balanced', random_state=42), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
y_pred_svm = svm_grid.predict(X_test_scaled)

# Evaluation - SVM
print("\n🔍 SVM Results:")
print("Best Params:", svm_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("F1-score (macro):", f1_score(y_test, y_pred_svm, average='macro'))
print("F1-score (micro):", f1_score(y_test, y_pred_svm, average='micro'))
print("F1-score (weighted):", f1_score(y_test, y_pred_svm, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# SVM Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.close()

# SVM Class Metrics Plot
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
svm_df = pd.DataFrame(svm_report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
svm_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title("SVM: Precision, Recall, F1-score per Class")
plt.tight_layout()
plt.savefig("svm_metrics_per_class.png")
plt.close()

# ---------- Random Forest ----------
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    rf_params, cv=5, scoring='accuracy'
)
rf_grid.fit(X_train, y_train)
y_pred_rf = rf_grid.predict(X_test)

# Evaluation - RF
print("\n🌳 Random Forest Results:")
print("Best Params:", rf_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1-score (macro):", f1_score(y_test, y_pred_rf, average='macro'))
print("F1-score (micro):", f1_score(y_test, y_pred_rf, average='micro'))
print("F1-score (weighted):", f1_score(y_test, y_pred_rf, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# RF Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()

# RF Class Metrics Plot
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
rf_df = pd.DataFrame(rf_report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
rf_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title("Random Forest: Precision, Recall, F1-score per Class")
plt.tight_layout()
plt.savefig("rf_metrics_per_class.png")
plt.close()

# Feature Importances (RF)
importances = rf_grid.best_estimator_.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:20], y=feat_imp.index[:20])
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importances_rf.png")
plt.close()

# Save the processed dataset
data.to_csv("processed_data.csv", index=False)
print("\n✅ Processed dataset saved as processed_data.csv")

