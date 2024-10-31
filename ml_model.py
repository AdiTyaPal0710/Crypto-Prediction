import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load the dataset
df = pd.read_excel('crypto.xlsx', sheet_name='CryptoData')

def train_model(data):
    # Define the feature and target columns
    features = [
        'Days_Since_High_Last_7_Days',
        '%_Diff_From_High_Last_7_Days',
        'Days_Since_Low_Last_7_Days',
        '%_Diff_From_Low_Last_7_Days'
    ]
    target = [
        '%_Diff_From_High_Next_5_Days',
        '%_Diff_From_Low_Next_5_Days'
    ]

    # Prepare the data (drop rows where any of the features or target are NaN)
    X = data[features].dropna()
    y = data[target].loc[X.index]  # Align y with X after dropping NaNs

    # Ensure there is no mismatch in X and y after dropping NaNs
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("No data available for training after dropping NaNs.")
        return None

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForestRegressor
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model1.fit(X_train, y_train)
    predictions1 = model1.predict(X_test)
    mse1 = mean_squared_error(y_test, predictions1)
    print(f"RandomForest MSE: {mse1}")

    # Initialize and train XGBRegressor
    model2 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model2.fit(X_train, y_train)
    predictions2 = model2.predict(X_test)
    mse2 = mean_squared_error(y_test, predictions2)
    print(f"XGBoost MSE: {mse2}")

    # Save both models
    joblib.dump(model1, "random_forest_regressor_crypto_model.pkl")
    joblib.dump(model2, "xgboost_crypto_model.pkl")

    # Return the model with the lower MSE
    best_model = model1 if mse1 < mse2 else model2
    joblib.dump(best_model, "best_crypto_model.pkl")  # Save best model for future predictions
    print(f"Best model saved with MSE: {min(mse1, mse2)}")
    return best_model

def predict_outcomes(input_data, model_choice="best"):
    # Load the model based on the choice
    if model_choice == "random_forest":
        model = joblib.load("random_forest_regressor_crypto_model.pkl")
    elif model_choice == "xgboost":
        model = joblib.load("xgboost_crypto_model.pkl")
    else:  # Default to the best model if no specific choice is made
        model = joblib.load("best_crypto_model.pkl")

    # Ensure the input data is in the correct format (as a 2D array)
    input_data = np.array(input_data).reshape(1, -1)
    return model.predict(input_data)

# Example usage
if __name__ == "__main__":
    # Train the model on the loaded data
    trained_model = train_model(df)
    
    # Predict outcomes for new input data (matching the feature order)
    new_input = [3, -2.5, 4, 1.5]  # Example input; adjust values as needed
    prediction = predict_outcomes(new_input)
    print(f"Predicted outcomes: {prediction}")
