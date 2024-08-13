from fastapi import FastAPI, HTTPException,UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Literal
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
from io import StringIO
from helpers import load_previous_dataset,load_latest_model


app = FastAPI()

# Loading models dynamically
lr_model,rf_model = load_latest_model()  

class ForecastRequest(BaseModel):
    forecast_horizon: int = 3
    model: str = "lr"

@app.post("/forecast/")
def forecast_energy_consumption(request: ForecastRequest):
    forecast_horizon = request.forecast_horizon
    model = request.model.lower()

    if model not in ["rf", "lr"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'random_forest' or 'linear_regression'.")

    combined_data = load_previous_dataset()
    last_data_point = combined_data.iloc[-1]

    forecast_values = []
    for _ in range(forecast_horizon):
        features = np.array([
            last_data_point['rolling_mean_2h'],
            last_data_point['rolling_mean_3h'],
            last_data_point['lag_1h'],
            last_data_point['lag_2h']
        ]).reshape(1, -1)

        if model == "rf":
            next_value = rf_model.predict(features)[0]
        elif model == "lr":
            next_value = lr_model.predict(features)[0]

        forecast_values.append(next_value)

        new_rolling_mean_2h = (last_data_point['y'] + next_value) / 2
        new_rolling_mean_3h = (last_data_point['y'] * 2 + next_value) / 3
        new_lag_1h = last_data_point['y']
        new_lag_2h = last_data_point['lag_1h']

        last_data_point = pd.Series({
            'y': next_value,
            'rolling_mean_2h': new_rolling_mean_2h,
            'rolling_mean_3h': new_rolling_mean_3h,
            'lag_1h': new_lag_1h,
            'lag_2h': new_lag_2h
        })

    forecast_index = pd.date_range(start=combined_data.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')

    forecast_results = []
    for n, value in zip(forecast_index, forecast_values):
        forecast_results.append({
            "hour": n.hour,
            "forecasted_value": value
        })

    return {"forecast": forecast_results}


@app.post("/train/")
async def train_model(file: UploadFile = File(...)):
    # Define the directory for saving the uploaded file
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the uploaded file to the data directory
    file_path = os.path.join(data_dir, 'energy_consumption_data.csv')
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load the dataset
    energy_data = pd.read_csv(file_path)
    
    # Assuming the 'timestamp' column exists and is used as index
    energy_data['timestamp'] = pd.to_datetime(energy_data['timestamp'])
    energy_data.set_index('timestamp', inplace=True)

    # Create Rolling Means and Lagged Features
    energy_data['rolling_mean_2h'] = energy_data['energy_consumption'].rolling(window=2).mean()
    energy_data['rolling_mean_3h'] = energy_data['energy_consumption'].rolling(window=3).mean()
    energy_data['lag_1h'] = energy_data['energy_consumption'].shift(1)
    energy_data['lag_2h'] = energy_data['energy_consumption'].shift(2)

    # Combine the Features into a Single DataFrame
    combined_data = pd.DataFrame({
        'original': energy_data['energy_consumption'],
        'rolling_mean_2h': energy_data['rolling_mean_2h'],
        'rolling_mean_3h': energy_data['rolling_mean_3h'],
        'lag_1h': energy_data['lag_1h'],
        'lag_2h': energy_data['lag_2h']
    }).dropna()

    # Train-Test Split
    train_size = int(len(combined_data) * 0.8)
    train_data = combined_data.iloc[:train_size]
    test_data = combined_data.iloc[train_size:]

    X_train = train_data[['rolling_mean_2h', 'rolling_mean_3h', 'lag_1h', 'lag_2h']]
    y_train = train_data['original']

    X_test = test_data[['rolling_mean_2h', 'rolling_mean_3h', 'lag_1h', 'lag_2h']]
    y_test = test_data['original']

    models_outputs = []

    # Create directory structure for saving models
    base_dir = './checkpoints'
    os.makedirs(base_dir, exist_ok=True)

    # Find the next available directory
    for i in range(1, 1000):
        dir_name = f"{base_dir}/v{i}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            break

    # Train the selected models and save them in the same directory
    for model_type in ['lr', 'rf']:
        if model_type == "lr":
            model = LinearRegression()
        elif model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Forecast using the Trained Model
        forecast = model.predict(X_test)

        # Calculate accuracy metrics
        mse = mean_squared_error(y_test, forecast)
        mae = mean_absolute_error(y_test, forecast)
        r2 = r2_score(y_test, forecast)

        # Save the model in the determined directory
        model_path = os.path.join(dir_name, f'{model_type}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Store the results
        models_outputs.append({
            "Model Type": model_type,
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            "R-squared": r2,
            "Model saved": True,
            "Directory": model_path
        })

    # Return the results
    return {"models_outputs": models_outputs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
