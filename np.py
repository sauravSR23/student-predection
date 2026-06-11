# ==========================================
# Student Marks Prediction ML Project
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==========================================
# Create Dataset
# ==========================================

data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9,10,5],
    "Attendance": [60,65,70,75,80,85,90,95,88,92,62,68,72,78,84,89,93,97,91,82],
    "Marks": [30,40,45,55,65,75,85,90,88,95,35,42,50,60,72,83,89,94,98,68]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

# ==========================================
# Data Cleaning
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ==========================================
# Data Visualization
# ==========================================

plt.scatter(df["Hours"], df["Marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours vs Marks")
plt.show()

# ==========================================
# Feature Selection
# ==========================================

X = df[["Hours", "Attendance"]]
y = df["Marks"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# Model Training
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

predictions = model.predict(X_test)

print("\nPredicted Marks:")
print(predictions)

# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)

# ==========================================
# New Student Prediction
# ==========================================

new_student = pd.DataFrame({
    "Hours": [6],
    "Attendance": [85]
})

result = model.predict(new_student)

print("\nPredicted Marks for New Student:", result[0])

# ==========================================
# Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": predictions
})

print("\nActual vs Predicted")
print(comparison)

# ==========================================
# End Project
# ==========================================