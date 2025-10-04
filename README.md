# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date:4/10/2025 
### Register Number:212223230105
### Name: K KESAVA SAI
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Clean_Dataset.csv', index_col=0)


result = adfuller(data['price'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]


lag_order = 13
model = AutoReg(train_data['price'], lags=lag_order)
model_fit = model.fit()


plt.figure(figsize=(10, 6))
plot_acf(data['price'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['price'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)


mse = mean_squared_error(test_data['price'], predictions)
print('Mean Squared Error (MSE):', mse)


plt.figure(figsize=(12, 6))
plt.plot(test_data['price'], label='Test Data - price')
plt.plot(predictions, label='Predictions - price', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()

```
### OUTPUT:
<img width="1232" height="641" alt="image" src="https://github.com/user-attachments/assets/0f09077e-3b2e-41a8-9c8f-c4b7e8e7463a" />
<img width="1069" height="569" alt="image" src="https://github.com/user-attachments/assets/aa5491f6-1c90-4865-989e-2f4940e371a4" />
<img width="1384" height="683" alt="image" src="https://github.com/user-attachments/assets/8c1e5668-832f-4399-b15f-22cef3acbe8e" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
