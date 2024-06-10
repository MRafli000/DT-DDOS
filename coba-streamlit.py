import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
import numpy as np

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Title of the Streamlit app
st.title("Decision Tree Classifier Application")

# Load the data from the CSV file
data = pd.read_csv('koreksi-6.csv')
st.write("Loaded Data:")
st.write(data)

st.write("-----------------------------------------------------------")

def labelencoder(data):
    for c in data.columns:
        if data[c].dtype == 'object':
            data[c] = data[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(data[c].values))
            data[c] = lbl.transform(data[c].values)
    return data

data1 = labelencoder(data)
data1.dropna(axis=0, inplace=True)

# Find missing values
missing_values = data1.isna().sum()
missing_values = missing_values.apply(lambda x: int(x))  # Convert to standard int
st.write("Missing Values:")
st.write(missing_values)

st.write("-----------------------------------------------------------")

# Encode categorical features
data = data.apply(LabelEncoder().fit_transform)

# Identify the target variable
target_variable = 'Event Name'

# Identify the features (columns) to be used for prediction
features = ['Low Level Category', 'Log Source', 'Event Count', 'Source IP', 'Source Port', 'Destination']

# Split the data into features (X) and target (y)
X = data[features]
y = data[target_variable]

# Check the mapping of the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(data['Event Name'])
label_mapping = {int(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
st.write("Label Mapping:")
st.write(label_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Slider for max_depth parameter of the decision tree
max_depth = st.slider("Select max_depth for Decision Tree", 1, 3, 3)
min_samples_leaf = st.slider("Select min_samples_leaf for Decision Tree", 1, 5, 5)

# Create a decision tree classifier with entropy criterion
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=min_samples_leaf)

# Perform 5-fold cross-validation
scores = cross_val_score(decision_tree, X_train, y_train, cv=5)
st.write(f"Cross-Validation Scores: {scores}")
st.write(f"Average Cross-Validation Score: {scores.mean():.2f}")

# Fit the model to the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = decision_tree.predict(X_test)

# Decode the numeric predictions back to original categorical values (optional)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_test_decoded = label_encoder.inverse_transform(y_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Test Accuracy: {accuracy:.2f}")

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-score: {f1:.2f}")

# Print the classification report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred, zero_division='warn'))

# Plot the confusion matrix with decoded labels
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay.from_estimator(decision_tree, X_test, y_test, display_labels=label_encoder.classes_, cmap='Blues', normalize='true', ax=ax)
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Create a correlation matrix
corr_matrix = data1.corr()

# Create a heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)

# Apply SelectKBest
selector = SelectKBest(f_regression, k=5)  # Select the top 5 features
X_new = selector.fit_transform(X, y)

# Print selected features
st.write("Selected Features Shape:")
st.write(X_new.shape)
