# Energy Demand Optimization Using Load Forecasting

This project analyzes hourly energy consumption data using Python and time series models to forecast demand and optimize production planning for large energy systems.

## 📌 Objective

To build a forecasting model using ARIMA for daily energy demand and simulate a 10% demand increase scenario to assess additional capacity needs.

## 🛠 Tools & Technologies

- Python (pandas, matplotlib, statsmodels, scikit-learn)
- Dataset: AEP Hourly Load (2004–2018) from PJM (via Kaggle)

## 🔍 Methods

- Converted hourly data to daily average
- Fitted ARIMA(5,1,0) model
- Calculated RMSE and MAPE to evaluate prediction
- Ran a scenario for January 2018 with +10% demand increase

## 📈 Results

- **RMSE:** 895.46 MW  
- **MAPE:** 4.74%  
- **Extra capacity needed for 10% rise in Jan 2018:** 1759.49 MW

## 📊 Sample Forecast Plot

![Forecast](forecast_plot.png)

## 📄 Files Included

- `arima_forecast.py`: Python code
- `forecast_plot.png`: Visualization of forecast vs real
- `report.pdf`: Full project report
- `scenario_analysis.txt`: Simulation output for January 2018
- `error_metrics.txt`: RMSE and MAPE

## 💡 Future Work

- Try SARIMA or multivariate models
- Integrate weather data (temperature, wind, etc.)
- Deploy model with real-time updates

---

### Prepared by: Bahar Işılar (Industrial Engineering Student)  
📅 Date: July 2025
