import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "square_feet": [750, 800, 850, 900, 950, 1000, 1100],
    "price": [150000, 160000, 170000, 180000, 190000, 200000, 220000]
}

df = pd.DataFrame(data)
X = df[["square_feet"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")



