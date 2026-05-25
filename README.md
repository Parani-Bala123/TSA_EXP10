# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 25/05/2026

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('/content/Walmart_Sales.csv')
data=data.head(200)
# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")

# Plot Weekly Sales Time Series
plt.plot(data['Date'], data['Weekly_Sales'])
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# Check stationarity
check_stationarity(data['Weekly_Sales'])

# ACF Plot
plot_acf(data['Weekly_Sales'])
plt.show()

# PACF Plot
plot_pacf(data['Weekly_Sales'])
plt.show()

# Split data into train and test
train_size = int(len(data) * 0.8)

train = data['Weekly_Sales'][:train_size]
test = data['Weekly_Sales'][train_size:]

# Build SARIMA Model
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

sarima_result = sarima_model.fit()

# Predictions
predictions = sarima_result.predict(
    start=len(train),
    end=len(train) + len(test) - 1
)

# RMSE Calculation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)

print('RMSE:', rmse)

# Plot Actual vs Predicted
plt.plot(data['Date'][train_size:], test, label='Actual')
plt.plot(data['Date'][train_size:], predictions, color='red', label='Predicted')

plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('SARIMA Model Predictions')

plt.legend()
plt.show()
```
### OUTPUT:
<img width="1030" height="721" alt="image" src="https://github.com/user-attachments/assets/aecbbfd1-d881-4fe8-9bf3-7c1d27535b8e" />

<img width="1002" height="550" alt="image" src="https://github.com/user-attachments/assets/09e0fa8e-e31a-4fc5-adc3-314840d70f96" />

<img width="1018" height="515" alt="image" src="https://github.com/user-attachments/assets/9f36d296-078c-458b-95b9-54f2787d2776" />

<img width="997" height="586" alt="image" src="https://github.com/user-attachments/assets/08a933b1-c0e2-4720-900b-9de378b22022" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
