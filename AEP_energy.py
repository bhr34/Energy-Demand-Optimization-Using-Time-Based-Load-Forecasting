import pandas as pd

# AEP_hourly.csv dosyasının tam yolunu buraya yaz
csv_path = r"C:\Users\Asus\Downloads\AEP_hourly.csv.zip"

# CSV'yi oku
df = pd.read_csv(csv_path, parse_dates=["Datetime"])

# İlk 5 satırı yazdır
print("Veri yüklendi. İlk 5 satır:")
print(df.head())

import matplotlib.pyplot as plt

# Saat bilgisi çıkar
df["Hour"] = df["Datetime"].dt.hour

# Ortalama saatlik enerji tüketimi
hourly_avg = df.groupby("Hour")["AEP_MW"].mean()

# Grafik çiz
plt.figure(figsize=(10,5))
hourly_avg.plot(kind="line", marker="o", color="green")
plt.title("Ortalama Saatlik Enerji Talebi (MW)")
plt.xlabel("Saat (0-23)")
plt.ylabel("Ortalama Talep (MW)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Ay bilgisini çıkar (1=Ocak, 12=Aralık)
df["Month"] = df["Datetime"].dt.month

# Aylık ortalama tüketimi hesapla
monthly_avg = df.groupby("Month")["AEP_MW"].mean()

# Grafik çiz
plt.figure(figsize=(10,5))
monthly_avg.plot(kind="bar", color="skyblue")
plt.title("Aylara Göre Ortalama Enerji Talebi")
plt.xlabel("Ay (1=Ocak, 12=Aralık)")
plt.ylabel("Ortalama Talep (MW)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np

# Tarihi sıraya koy
df = df.sort_values("Datetime")

# Günlük ortalama talep (Datetime'ı tarihe indirgeriz)
df["DateOnly"] = df["Datetime"].dt.date
daily_avg = df.groupby("DateOnly")["AEP_MW"].mean().reset_index()

# X: gün sırası (0,1,2,3,...), y: talep
X = np.arange(len(daily_avg)).reshape(-1, 1)
y = daily_avg["AEP_MW"].values

# Modeli kur
model = LinearRegression()
model.fit(X, y)

# Tahmin et (orijinal + 7 gün sonrası)
X_pred = np.arange(len(daily_avg)+7).reshape(-1,1)
y_pred = model.predict(X_pred)

# Grafikle göster
plt.figure(figsize=(10,5))
plt.plot(daily_avg["DateOnly"], y, label="Gerçek", color="blue")
plt.plot(pd.date_range(daily_avg["DateOnly"].iloc[0], periods=len(y_pred)), y_pred, label="Tahmin", color="orange", linestyle="--")
plt.title("Enerji Talebi: Gerçek vs Lineer Tahmin")
plt.xlabel("Tarih")
plt.ylabel("Talep (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import zipfile

# ZIP dosyasının yolu
zip_path = r"C:\Users\Asus\Downloads\AEP_hourly.csv.zip"

# ZIP içindeki CSV dosyasını aç
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # ZIP içindeki ilk dosya adını al
    csv_name = zip_ref.namelist()[0]
    with zip_ref.open(csv_name) as csv_file:
        df = pd.read_csv(csv_file)

# Tarih sütununu datetime'a çevir
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Datetime'ı indeks yap
df.set_index('Datetime', inplace=True)

# İlk 5 satırı göster
print("İlk 5 veri satırı:")
print(df.head())

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Günlük ortalama al (saatlikten)
df_daily = df.resample('D').mean()

# Hedef veri
y = df_daily['AEP_MW']

# ARIMA(5,1,0) modeli kur
model = ARIMA(y, order=(5, 1, 0))
model_fit = model.fit()

# Son 1 yılı tahmin et
forecast = model_fit.predict(start=len(y)-365, end=len(y)-1, typ='levels')

# Grafikle göster
plt.figure(figsize=(12, 6))
plt.plot(y[-365:], label='Actual Demand', color='blue')
plt.plot(forecast, label='ARIMA Forecast', color='orange', linestyle='--')
plt.title('Energy Demand Forecast using ARIMA (Last 1 Year)')
plt.xlabel('Date')
plt.ylabel('Demand (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Gerçek ve tahmin değerleri
y_true = y[-365:]
y_pred = forecast

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAPE
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

# Sonuçları yazdır
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} MW")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

# Ocak 2018 verilerini al
january_data = y['2018-01-01':'2018-01-31']
january_mean = january_data.mean()

# %10 artış senaryosu
target_mean = january_mean * 1.10
increase_needed = target_mean - january_mean

print(f"January 2018 Average Demand: {january_mean:.2f} MW")
print(f"Target with 10% Increase: {target_mean:.2f} MW")
print(f"Additional Capacity Needed: {increase_needed:.2f} MW")
