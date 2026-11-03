# DATA-PREPROCESSING
# AIM
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 
# Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas. 
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply Standard Scaler for feature scaling. 
13.	Display the preprocessed training and testing datasets. 

# Program
# Step 1: Import the required Python libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Step 2: Mount Google Drive (if running in Google Colab) and load the dataset
from google.colab import drive
drive.mount('/content/drive')

# Replace 'Data.csv' with your actual dataset path in Google Drive
dataset_path = '/content/drive/My Drive/Data.csv'
df = pd.read_csv(dataset_path)

# Step 3: Display the first few records of the dataset
print("--- First 5 Records ---")
print(df.head())

# Step 4: Inspect the dataset using df.info() and df.shape
print("\n--- Dataset Info ---")
df.info()

print(f"\n--- Dataset Shape (Rows, Columns) ---")
print(df.shape)

# Step 5: Separate the independent variables (X) and dependent variable (Y)
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column ('Purchased')

# Step 6: Convert the independent variables into an array (already done via .values, let's verify)
print("\n--- Independent Variables (X) Before Preprocessing ---")
print(X)

# Step 7: Identify and handle missing values using SimpleImputer with the mean strategy
# Imputing missing values for numeric columns (Age and Salary, indices 1 and 2)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Step 8: Encode the categorical Country column using LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Step 9: Apply One-Hot Encoding to convert categorical country values into dummy variables
# Note: Country is at index 0
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

# Step 10: Encode the dependent variable Purchased using LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Step 11: Split the dataset into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 12: Apply Standard Scaler for feature scaling
# Scale only the numeric features (excluding the one-hot encoded dummy variables for country)
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# Step 13: Display the preprocessed training and testing datasets
print("\n--- X_train (Preprocessed) ---")
print(X_train)

print("\n--- X_test (Preprocessed) ---")
print(X_test)

print("\n--- y_train ---")
print(y_train)

print("\n--- y_test ---")
print(y_test)

# RESULT
The dataset was successfully loaded, inspected, and preprocessed.
