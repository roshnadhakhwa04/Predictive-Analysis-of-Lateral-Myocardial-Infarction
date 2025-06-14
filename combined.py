# EDA and Model Training Pipeline with Hyperparameter Tuning

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
data = pd.read_csv("merged_df.csv")

# Step 2: Perform initial EDA
print("\n📊 Dataset Info:")
print(data.info())
print("\n🔍 First 5 Rows:")
print(data.head())

print("\n❓ Missing Values:")
print(data.isnull().sum())

# Visualize missing values
msno.matrix(data)
plt.title("Missing Values Matrix")
plt.tight_layout()
plt.savefig("missing_values_matrix.png")
plt.close()

# Step 3: Visualize distributions
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns

for col in num_cols:
    plt.figure()
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"dist_{col}.png")
    plt.close()

# Boxplots
for col in num_cols:
    plt.figure()
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"box_{col}.png")
    plt.close()

# Step 4: Handle missing values
for col in num_cols:
    if data[col].isnull().sum() > 0:
        skew = data[col].skew()
        fill_value = data[col].median() if abs(skew) > 1 else data[col].mean()
        data[col].fillna(fill_value, inplace=True)

for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Step 5: Correlation heatmap
corr_matrix = data[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Feature_Correlation_Heatmap.png")
plt.close()

# Step 6: Drop highly correlated features
threshold = 0.6
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
data.drop(columns=to_drop, inplace=True)

# Step 7: Violin plots for numeric features vs target
if 'lat_im' in data.columns:
    for col in num_cols:
        if col != 'lat_im':
            plt.figure(figsize=(8, 4))
            sns.violinplot(x='lat_im', y=col, data=data)
            plt.title(f"{col} distribution by lat_im")
            plt.tight_layout()
            plt.savefig(f"violin_{col}.png")
            plt.close()

# Step 8: Count plots for categorical features
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=col, hue='lat_im')
    plt.title(f"Count of {col} by lat_im")
    plt.tight_layout()
    plt.savefig(f"count_{col}.png")
    plt.close()

# Step 9: Train-Test split
X = data.drop(columns='lat_im')
y = data['lat_im']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 10: Train and evaluate SVM with GridSearchCV
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)

print(f"\n🔍 Best SVM Params: {svm_grid.best_params_}")
y_pred_svm = svm_grid.predict(X_test_scaled)

print("\n🔍 SVM Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.close()

# Step 11: Train and evaluate Random Forest with GridSearchCV
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)

print(f"\n🌳 Best RF Params: {rf_grid.best_params_}")
y_pred_rf = rf_grid.predict(X_test)

print("\n🌳 Random Forest Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("RF_confusion_matrix.png")
plt.close()

# Random Forest Feature Importances
importances = rf_grid.best_estimator_.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values[:20], y=feat_imp.index[:20])
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importances_rf.png")
plt.close()

# Save processed dataset
data.to_csv("processed_data.csv", index=False)
print("\n✅ Processed dataset saved as processed_data.csv")

