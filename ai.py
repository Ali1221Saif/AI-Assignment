import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")  # Replace with your data path

# Handle missing values and non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object']).columns
numeric_cols = data.select_dtypes(exclude=['object']).columns

# Impute missing values for numeric columns with median
numeric_imputer = SimpleImputer(strategy="median")
data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=non_numeric_cols)

# Separate features and target
X = data.drop(columns=['stroke'])
y = data['stroke']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Nearest Neighbor
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# Evaluate performance
print("Decision Tree:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Precision:", precision_score(y_test, dt_predictions))
print("Recall:", recall_score(y_test, dt_predictions))
print("F1-score:", f1_score(y_test, dt_predictions))

print("\nNearest Neighbor:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Precision:", precision_score(y_test, knn_predictions))
print("Recall:", recall_score(y_test, knn_predictions))
print("F1-score:", f1_score(y_test, knn_predictions))