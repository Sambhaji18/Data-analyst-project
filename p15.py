#Case study 1: Sredicting House Prices Using linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = {
    "Size (sq ft)": [1500, 1700, 2000, 2400, 3000],
    "Price (in $)": [300000, 350000, 400000, 500000, 600000]
}
df = pd.DataFrame(data)

# Step 2: Split the data into features (X) and target (y)
X = df[["Size (sq ft)"]]
y = df["Price (in $)"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 7: Visualize the results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price (in $)")
plt.title("Linear Regression - House Prices")
plt.legend()
plt.show()

# Predict price for a new house size
new_size = pd.DataFrame([[2100]], columns=["Size (sq ft)"])  # Size in sq ft
predicted_price = model.predict(new_size)
print(f"Predicted price for a house of size {new_size.iloc[0, 0]} sq ft: ${predicted_price[0]:,.2f}")

#case study 2: Predicting Student Scores Based on Study Hours Using Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = {
    "Study Hours": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "Score": [30, 35, 50, 65, 70, 75, 85, 90]
}
df = pd.DataFrame(data)

# Step 2: Split the data into features (X) and target (y)
X = df[["Study Hours"]]
y = df["Score"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 7: Visualize the results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Score (%)")
plt.title("Linear Regression - Study Hours vs Score")
plt.legend()
plt.show()

# Predict score for a new study duration
new_hours = pd.DataFrame([[4.2]], columns=["Study Hours"])  # Hours of study
predicted_score = model.predict(new_hours)
print(f"Predicted score for {new_hours.iloc[0, 0]} study hours: {predicted_score[0]:.2f}%")
