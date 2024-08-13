# Energy Consumption Forecasting

This project is designed to forecast energy consumption using machine learning models like Linear Regression and Random Forest. The project includes a Python script that processes the data, trains the models, and saves them for future use. The environment is containerized using Docker, allowing for easy deployment and reproducibility.

## Project Structure


├── Dockerfile
├── README.md
├── requirements.txt
├── energy_consumption_data.csv
├── forecast_energy.py
└── saved_models
├── linear_regression_model.pkl
└── random_forest_model.pkl


- `Dockerfile`: Instructions to build the Docker image.
- `README.md`: Documentation for the project.
- `requirements.txt`: Python dependencies required to run the project.
- `energy_consumption_data.csv`: Dataset used for training the models.
- `forecast_energy.py`: Python script that processes the data, trains the models, and makes predictions.
- `saved_models/`: Directory where trained models are saved as `.pkl` files.

## Prerequisites

Before you begin, ensure you have Docker installed on your system. You can download Docker from [here](https://www.docker.com/products/docker-desktop).

## Installation


**Build the Docker image:**

Run the following command in the terminal to build the Docker image:

  ```bash
  docker build -t energy-forecasting-image .
  ```

## Usage

**Running the Docker container:**

After building the Docker image, run the following command to start the container and execute the Python script:

  ```bash
  docker run -v .:/app -p 8001:8001 energy-forecasting-image
  ```

  This command will:

  Load the dataset. 

  Forecast energy consumption using the trained models.

  Display the results including plots and evaluation metrics.


## Endpoints

### 1. `/train/` [POST]
This endpoint allows you to upload a CSV file containing energy consumption data, which is used to train two models: Linear Regression (LR) and Random Forest Regressor (RF). The trained models are saved for later use in forecasting.

- **Request:**
  - **File:** A CSV file with at least two columns:
    - `timestamp`: The date and time of the energy consumption measurement.
    - `energy_consumption`: The amount of energy consumed.

- **Process:**
  1. The uploaded file is saved to the `data` directory.
  2. The dataset is processed to create rolling means and lagged features.
  3. The dataset is split into training and testing sets.
  4. Linear Regression and Random Forest models are trained on the data.
  5. The models are saved in a newly created directory under `checkpoints`.

- **Response:**
  - A JSON object containing:
    - `Model Type`: The type of model (`lr` or `rf`).
    - `Mean Squared Error`: The mean squared error of the model on the test set.
    - `Mean Absolute Error`: The mean absolute error of the model on the test set.
    - `R-squared`: The R-squared value of the model on the test set.
    - `Model saved`: A boolean indicating whether the model was saved.
    - `Directory`: The directory where the model was saved.

- **Example Request:**
  ```bash
  curl -X POST "http://localhost:8001/train/" -F "file=@path_to_your_file/energy_consumption_data.csv"
  ```

### 2. `/forecast/` [POST]
This endpoint allows you to forecast future energy consumption based on the last trained models.

**Request:**

- **JSON Body:**
  - `forecast_horizon`: The number of hours into the future you want to forecast (default is 3).
  - `model`: The model to use for forecasting (`lr` for Linear Regression, `rf` for Random Forest; default is `lr`).

**Process:**

1. The endpoint loads the last trained models.
2. The previous dataset used for training is loaded.
3. Forecasts are generated for the specified number of hours using the chosen model.

**Response:**

- A JSON object containing a list of forecasted values for the specified hours.

**Example Request:**

```bash
curl -X POST "http://localhost:8001/forecast/" -H "Content-Type: application/json" -d '{"forecast_horizon": 3, "model": "lr"}'
