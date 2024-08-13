import joblib
import os
import pandas as pd



# Function to load the latest model dynamically
def load_latest_model(base_dir='/app/checkpoints'):
    folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))], reverse=True)
    last_folder = folders[0] if folders else None
    if last_folder:
        lr_model_path = os.path.join(base_dir, last_folder, 'lr_model.pkl')
        lr_model = joblib.load(lr_model_path)
        rf_model_path = os.path.join(base_dir, last_folder, 'rf_model.pkl')
        rf_model = joblib.load(rf_model_path)
        return lr_model,rf_model
    else:
        raise FileNotFoundError("No model found in the checkpoints directory.")

# Loading models dynamically
lr_model,rf_model = load_latest_model()  


def load_previous_dataset(dir_name='/app/data/energy_consumption_data.csv'):
    data = pd.read_csv(dir_name)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.rename(columns={'timestamp': 'ds', 'energy_consumption': 'y'})
    data.set_index('ds', inplace=True)

    data['rolling_mean_2h'] = data['y'].rolling(window=2).mean().shift(1)
    data['rolling_mean_3h'] = data['y'].rolling(window=3).mean().shift(1)
    data['lag_1h'] = data['y'].shift(1)
    data['lag_2h'] = data['y'].shift(2)

    # Drop NaN values resulting from the shifts
    data = data.dropna()

    return data