import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Download the latest version of the dataset
path = kagglehub.dataset_download("arshid/iris-flower-dataset")
print("Path to dataset files:", path)

# Load the Iris dataset from CSV
iris_df = pd.read_csv(f'{path}/Iris.csv')  # Adjust the filename if necessary

# Display the first few rows of the dataset
print(iris_df.head())

# Define features and target variable
X = iris_df.drop('species', axis=1)  # Assuming 'species' is the target column
y = iris_df['species']

# Visualize the dataset
sns.pairplot(iris_df, hue='species')
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))
