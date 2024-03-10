import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier
from EDA import myEDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

dataset_path = ".//Iris_Data.csv"
seedata = myEDA(dataset_path)
print(seedata.df)
# seedata.Data_profiling()
# %matplotlib qt5
seedata.heatmap()

# --- Seperating Dependent Features ---
x = seedata.df2
y = seedata.df["species"]
print(x)
print(y)
# --- Splitting Dataset ---
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(x_train)

# Define the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Create a pipeline with scaler and KNN classifier
pipe = Pipeline([("scaler", StandardScaler()), ("algo", knn)])

# Define the hyperparameter grid
parameter_knn = {
    "algo__n_neighbors": [2, 5, 10, 17],
    "algo__leaf_size": [1, 10, 11, 30],
}
model = GridSearchCV(pipe, parameter_knn, cv=10, n_jobs=-1, verbose=1)
