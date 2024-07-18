import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import RocCurveDisplay, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

AnaemiaData = pd.read_csv("../Dataset/anaemia_dataset.csv")

# Renaming columns
cols_to_rename = {'%Red Pixel': 'red_pixel',
                  '%Green pixel': 'green_pixel',
                  '%Blue pixel': 'blue_pixel'}

AnaemiaData.rename(columns=cols_to_rename, inplace=True)
AnaemiaData.columns = map(str.lower, AnaemiaData.columns)

# Removing column which do not need in the dataset
AnaemiaData.drop("number", axis=1, inplace=True)

# Standardising the values in the sex category
AnaemiaData['sex'] = AnaemiaData['sex'].replace(['M ', 'F '],['M','F'])

# Selecting our X and y
X = AnaemiaData.drop('anaemic', axis=1)
y = AnaemiaData['anaemic']

# Splitting data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Columns to be scaled
numeric_features = ["red_pixel", "green_pixel", "blue_pixel","hb"]

# Column to be binned and one-hot encoded
categorical_features = ["sex"]

# Columns to be scaled
numeric_features = ["red_pixel", "green_pixel", "blue_pixel","hb"]

# Column to be binned and one-hot encoded
categorical_features = ["sex"]

# Columns to be scaled
numeric_features = ["red_pixel", "green_pixel", "blue_pixel","hb"]

# Column to be binned and one-hot encoded
categorical_features = ["sex"]

# Creating transformers
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

# Combining all transformers into a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)], remainder='passthrough')

# Creating and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())]).set_output(transform='pandas')

# Training the classifier on the training data
pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:,1]

# Calculating accuracy
acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc_score}") # Accuracy: 0.9687

# save the model to disk
joblib.dump(pipeline, "model/log_reg_model.sav")