## Mercedes Price Prediction with Neural Network

This project predicts used Mercedes car prices using a deep learning regression model built with TensorFlow/Keras. It demonstrates a complete end-to-end machine learning workflow including data analysis, preprocessing, model training, evaluation, visualization, and interactive user-based prediction.

## 📌 Project Pipeline

The workflow starts with Exploratory Data Analysis (EDA) to understand feature distributions and relationships. To reduce skewness and improve model stability, the top 1% of price values were removed as outliers. Categorical variables were transformed using one-hot encoding and all numerical features were scaled to the 0–1 range with MinMaxScaler.

A fully connected neural network with the architecture 64 → 32 → 16 → 1 was trained using the Adam optimizer and Mean Squared Error (MSE) loss function. EarlyStopping (patience = 25) was applied to prevent overfitting and stop training when validation loss stopped improving.

The model was evaluated using MAE, RMSE, and R² metrics and visualized with loss curves and real vs predicted price scatter plots. The final model supports interactive price prediction based on user input.

## 📊 Dataset

Dataset: merc.csv
The dataset contains features such as:

year

mileage

fuel type

transmission

other vehicle attributes

These features are used to predict the car price.

## 🧠 Model Details

Architecture:
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1)

Loss: Mean Squared Error (MSE)

Optimizer: Adam

Regularization: EarlyStopping (patience = 25)

## 📈 Model Performance

MAE: ~2255

RMSE: ~3000

R² Score: Strong correlation between real and predicted prices

## 📷 Visualizations

The project includes:

Price distribution (before/after outlier removal)

Correlation heatmap

Mileage vs price relationship

Training vs validation loss curve

Real vs predicted price scatter plot

## 🧪 User Input Prediction

After training, the model asks for:

Car year

Mileage

Fuel type

Transmission

and returns a predicted price.

## ▶️ How to Run

1. Place `merc.csv` in the same folder as `mercedes_price_prediction.py`  
2. Install the required libraries  
3. Run the script

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
python mercedes_price_prediction.py
```
