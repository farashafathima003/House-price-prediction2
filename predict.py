
import pickle
import pandas as pd

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Enter house details
house = pd.DataFrame([{
    "area": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "age": 5
}])

# Predict the price
price = model.predict(house)[0]

print("House Details")
print("Area: 1200 square feet")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Age: 5 years")

print("\nPredicted House Price:")
print("Rs.", round(price, 2))
