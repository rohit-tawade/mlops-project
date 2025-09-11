# Databricks notebook source
# MAGIC %md
# MAGIC 🔹 Step 1 – Upload Dataset (Iris CSV) to DBFS

# COMMAND ----------

import requests

# Download iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
response = requests.get(url)

# Save under your user workspace (private DBFS path)
path = "/Workspace/Repos/rohit.s.tawade@gmail.com/mlops-project/Data/iris.csv"

with open(path, "w") as f:
    f.write(response.text)

print("✅ Saved to:", path)


# COMMAND ----------

# MAGIC %md
# MAGIC 🔹 Step 2 – Load Data & Do Basic EDA

# COMMAND ----------

# MAGIC %md
# MAGIC Load with Pandas

# COMMAND ----------

import pandas as pd

df = pd.read_csv("/Workspace/Users/rohit.s.tawade@gmail.com/iris.csv")
print(df.head())
print(df.describe())


# COMMAND ----------

# MAGIC %md
# MAGIC Plot some charts

# COMMAND ----------

import matplotlib.pyplot as plt
df.plot(kind="scatter", x="sepal_length", y="sepal_width")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 🔹Step 3 – Train a Simple Model

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load and prepare data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Train/test split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "/Workspace/Repos/rohit.s.tawade@gmail.com/mlops-project/Model/iris_model.pkl")


# COMMAND ----------

# MAGIC %md
# MAGIC 🔹Step 4 MLflow Tracking

# COMMAND ----------

import mlflow
import mlflow.sklearn
mlflow.set_experiment("/Users/rohit.s.tawade@gmail.com/iris_experiment")
