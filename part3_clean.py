import pandas as pd
import joblib

print("Loading troop_movements_1m.csv")
df = pd.read_csv("troop_movements_1m.csv")

print("Columns in the dataset:", df.columns)

df["unit_type"] = df["unit_type"].replace("invalid_unit", "unknown")
df["location_x"] = df["location_x"].fillna(method="ffill")  
df["location_y"] = df["location_y"].fillna(method="ffill")


df.to_parquet("troop_movements_1m.parquet", engine="pyarrow")
print("Cleaned data saved as troop_movements_1m.parquet")

print("Loading trained model...")
model = joblib.load("data/trained_model.pkl") 
print("Model loaded successfully")

print("Preparing data for prediction")
X = df[["unit_type", "homeworld"]]  
X_encoded = pd.get_dummies(X)
X_encoded = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)


print("Making predictions")
predictions = model.predict(X_encoded)


df["prediction"] = predictions
df.to_csv("predicted_results.csv", index=False)
print("Predictions saved to predicted_results.csv")
