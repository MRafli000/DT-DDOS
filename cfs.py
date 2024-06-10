import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load the data from the CSV file
data = pd.read_csv('FINAL.csv')
print(data)

print("-----------------------------------------------------------")

# Encode categorical features
data = data.apply(LabelEncoder().fit_transform)

# Identify the target variable
target_variable = 'Event Name'

# Identify the features (columns) to be used for prediction
features = ['Low Level Category', 'Log Source', 'Time', 'Source IP', 'Event Count', 'Source Port', 'Destination']

# Split the data into features (X) and target (y)
X = data[features]
y = data[target_variable]

# Check the mapping of the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(data['Event Name'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Apply CFS (Correlation-based Feature Selection)
correlation_matrix = data.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Select features that have high correlation with the target and low correlation with each other
selected_features = []
for feature in features:
    if abs(correlation_matrix[feature][target_variable]) > 0.1:  # Threshold for feature-target correlation
        selected_features.append(feature)
print("Selected Features after CFS:", selected_features)

X_selected = data[selected_features]

# Split the data into training and testing sets with selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Create a decision tree classifier with entropy criterion and no max depth limit
decision_tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=4)  # No max_depth limit

# Perform 3-fold stratified cross-validation with reduced n_splits
stratified_kfold = StratifiedKFold(n_splits=3)  # Reduce n_splits to 3 to avoid warning
scores = cross_val_score(decision_tree, X_train, y_train, cv=stratified_kfold)
print(f"Cross-Validation Scores: {scores}")
print(f"Average Cross-Validation Score: {scores.mean():.2f}")

# Fit the model to the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = decision_tree.predict(X_test)

# Decode the numeric predictions back to original categorical values (optional)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_test_decoded = label_encoder.inverse_transform(y_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Print the classification report with decoded labels
print("Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, zero_division='warn'))

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

# Plot the heatmap for the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Create a correlation matrix
corr_matrix = data.corr()

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
