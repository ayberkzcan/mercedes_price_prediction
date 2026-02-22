import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

"""
Mercedes Price Prediction with Neural Network

Pipeline:
- Exploratory Data Analysis
- Outlier Removal (Top 1%)
- One-Hot Encoding
- MinMax Scaling
- Neural Network (64-32-16)
- EarlyStopping
- Evaluation (MAE, RMSE, R2)
- User Input Prediction

Dataset: merc.csv
"""

df = pd.read_csv("merc.csv")

# Basic data inspection
print(df.head(), "\n")
print(df.describe(), "\n")
print(df.isnull().sum(), "\n")

plt.figure(figsize=(8,5))
sbn.histplot(df["price"], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,5))
sbn.countplot(x="year", data=df)
plt.title("Yıllara Göre Araç Sayısı")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,7))
numeric_df = df.select_dtypes(include=[np.number])
sbn.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(8,5))
sbn.scatterplot(x="mileage", y="price", data=df)
plt.title("Mileage - Price")
plt.show()

# Remove top 1% price outliers to reduce skewness
print(df.sort_values("price",ascending=False).head(20).reset_index(drop=True))
upper_limit = df["price"].quantile(0.99)
newdf = df[df["price"] <= upper_limit]
newdf = newdf.reset_index(drop=True)
print(newdf.describe())

plt.figure(figsize=(8,5))
sbn.histplot(newdf["price"], bins=50, kde=True)
plt.title("Updated Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sbn.scatterplot(x="mileage", y="price", data=newdf)
plt.title("Mileage - Price (Outliers Removed)")
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sbn.histplot(df["price"])
plt.title("Before Outlier")
plt.subplot(1,2,2)
sbn.histplot(newdf["price"])
plt.title("After Outlier")
plt.show()

print(newdf.groupby("year")["price"].mean())
newdf = newdf[newdf.year != 1970]
print(newdf.groupby("year")["price"].mean())
newdf = pd.get_dummies(newdf, drop_first=True)

# Separate features and target variable
y = newdf["price"].values
X = newdf.drop("price", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(len(x_train))
print(len(x_test))

# Scale features to 0-1 range for neural network
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)

# Build neural network model
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Stop training when validation loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=25,
    verbose=0
)

model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test,y_test),
    batch_size=250,
    epochs=300,
    callbacks=[early_stop],
    verbose=0
)
print(f"Training stopped at epoch: {len(model.history.history['loss'])}")

lossdata = pd.DataFrame(model.history.history)
lossdata.plot(figsize=(8,5))
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.show()

predictions = model.predict(x_test).flatten()

print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions):.3f}")

plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions, alpha=0.6)
max_val = max(y_test.max(), predictions.max())
min_val = min(y_test.min(), predictions.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r')
plt.xlabel("Real Price")
plt.ylabel("Predicted Price")
plt.title("Real vs Predicted Price")
plt.show()

feature_columns = newdf.drop("price", axis=1).columns.tolist()  # 37 feature

year = int(input("Car year: "))
mileage = float(input("Mileage (km): "))
fuel = input("Fuel type (diesel/gasoline/hybrid/other): ").lower()
trans = input("Transmission type (manual/automatic): ").lower()

user_data = dict.fromkeys(feature_columns, 0)

user_data["year"] = year
user_data["mileage"] = int(mileage)
fuel_col = f"fuelType_{fuel}"
trans_col = f"transmission_{trans}"

if fuel_col in user_data:
    user_data[fuel_col] = 1
if trans_col in user_data:
    user_data[trans_col] = 1

input_df = pd.DataFrame([user_data])
input_scaled = scaler.transform(input_df.values)
predicted_price = model.predict(input_scaled).flatten()[0]

print(f"Predicted Price: {predicted_price:.2f}")