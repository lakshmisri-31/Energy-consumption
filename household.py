#-->>Importing needed Libraries
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV





#-->>File Extraction
#defining the file path
zip_file_path = r'C:\Users\laksh\Downloads\individual+household+electric+power+consumption.zip'
extract_path = r'C:\Users\laksh\Downloads\household_power_consumption'

#extracting the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

#loading the dataset
csv_file_path = extract_path + r'\household_power_consumption.txt'  # Update based on actual file
data = pd.read_csv(csv_file_path, sep=';', low_memory=False)

#displaying dataset structure
print(data.info())
print(data.head())





#-->>Data Preprocessing
#handling of missing values
data.replace('?', pd.NA, inplace=True)

#converting numeric columns to appropriate types
numeric_columns = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2'
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#checking for missing values
print(data.isnull().sum())





#-->Date and Time Parsing
#ensuring 'Time' is a string before concatenation with 'Date'
data['Datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='coerce', dayfirst=True)

#extracting date and time component
data['Date'] = data['Datetime'].dt.date
data['Time'] = data['Datetime'].dt.time
data['Year'] = data['Datetime'].dt.year
data['Month'] = data['Datetime'].dt.month
data['Day'] = data['Datetime'].dt.day
data['Hour'] = data['Datetime'].dt.hour
data['Minute'] = data['Datetime'].dt.minute
data['Second'] = data['Datetime'].dt.second

#checking the updated dataframe
print(data.head())





#-->handling missing data
#droping rows with missing Datetime
data.dropna(subset=['Datetime'], inplace=True)

#selecting only numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns

#filling missing numeric values with column mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

#confirming no missing values remain
print(data.isnull().sum())





#-->>Feature Engineering
#adding new columns for date, hour, and day of the week
data['Date'] = data['Datetime'].dt.date
data['Hour'] = data['Datetime'].dt.hour
data['DayOfWeek'] = data['Datetime'].dt.day_name()

#calculating daily averages for energy consumption
daily_avg = data.groupby('Date').agg({
    'Global_active_power': 'mean',
    'Global_reactive_power': 'mean',
    'Voltage': 'mean',
    'Global_intensity': 'mean'
}).reset_index()
daily_avg.rename(columns=lambda x: f'Daily_{x}' if x != 'Date' else x, inplace=True)

#adding rolling averages for energy consumption (e.g., 1-day and 7-day rolling averages)
data['Rolling_1Day'] = data['Global_active_power'].rolling(window=24).mean()
data['Rolling_7Day'] = data['Global_active_power'].rolling(window=24 * 7).mean()

#identifying peak usage times by finding the hour with the maximum average usage
hourly_avg = data.groupby('Hour').agg({'Global_active_power': 'mean'}).reset_index()
peak_hour = hourly_avg.loc[hourly_avg['Global_active_power'].idxmax(), 'Hour']
print(f"\nPeak usage hour based on average consumption: {peak_hour}:00")

#adding a flag for peak usage hours
data['IsPeakHour'] = data['Hour'].apply(lambda x: 1 if x == peak_hour else 0)

#summarizing energy usage by day of the week
weekly_avg = data.groupby('DayOfWeek').agg({
    'Global_active_power': 'mean',
    'Global_reactive_power': 'mean'
}).reset_index()

#adding total energy consumed/day
data['DailyTotalEnergy'] = data.groupby('Date')['Global_active_power'].transform('sum')

scaler = MinMaxScaler()
data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']] = scaler.fit_transform(
    data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
)

#displaying the first few rows of the updated dataset
print("\nFeature-Engineered Dataset Preview:")
print(data.head())

#saving the feature-engineered dataset for later use
data.to_csv('feature_engineered_data.csv', index=False)





#-->>Exploratory Data Analysis (EDA)
#ensuring that 'datetime' is in the correct datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

#filtering out only numeric columns(float64 and int64)
numeric_data = data.select_dtypes(include=['float64', 'int64'])

#1.correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

#2.distribution analysis(histograms)
energy_vars = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
data[energy_vars].hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Energy Consumption Variables", fontsize=16)
plt.show()

#3.time series trends

#daily trend
data_daily = data.set_index('Datetime').resample('D').mean(numeric_only=True)
plt.figure(figsize=(14, 6))
plt.plot(data_daily.index, data_daily['Global_active_power'], label='Daily Avg Global Active Power', color='blue')
plt.title("Daily Trend of Global Active Power")
plt.xlabel("Date")
plt.ylabel("Global Active Power (normalized)")
plt.legend()
plt.grid(True)
plt.show()

#monthly trend
data['MonthYear'] = data['Datetime'].dt.to_period('M')
monthly_avg = data.groupby('MonthYear').mean(numeric_only=True)
plt.figure(figsize=(14, 6))
plt.plot(monthly_avg.index.to_timestamp(), monthly_avg['Global_active_power'], label='Monthly Avg Global Active Power', color='green')
plt.title("Monthly Trend of Global Active Power")
plt.xlabel("Month")
plt.ylabel("Global Active Power (normalized)")
plt.legend()
plt.grid(True)
plt.show()

#yearly trend
data['Year'] = data['Datetime'].dt.year
yearly_avg = data.groupby('Year').mean(numeric_only=True)
plt.figure(figsize=(10, 6))
plt.bar(yearly_avg.index, yearly_avg['Global_active_power'], color='orange', edgecolor='black')
plt.title("Yearly Trend of Global Active Power")
plt.xlabel("Year")
plt.ylabel("Global Active Power (normalized)")
plt.grid(True)
plt.show()





#-->>Modeling
#->Linear Regression
#replacing 'Global_active_power' with the target variable
X = data[['Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]  # Example features
y = data['Global_active_power']

#train-test split(80%,20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#prediction on the test set
y_pred = lr_model.predict(X_test)

#evaluating the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Baseline Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-Squared (R²): {r2:.4f}")





#->Random Forest Regressor
#initializing the random forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20)

#training the model
rf_model.fit(X_train, y_train)

#predictions
y_pred_rf = rf_model.predict(X_test)

#evaluating the random forest model
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"R-Squared (R²): {r2_rf:.4f}")





#-->>Feature Importance
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

#plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()





#-->>Hyperparameter gird
#defining hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#randomized search
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid,
                                   n_iter=20, cv=3, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

#best parameters and model
best_rf_model = random_search.best_estimator_
print("\nBest Random Forest Parameters:", random_search.best_params_)

#evaluating tuned random forest
y_pred_best_rf = best_rf_model.predict(X_test)
rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
print(f"Best Tuned RF RMSE: {rmse_best_rf:.4f}")





#-->>Visualization of Predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best_rf, alpha=0.7, color='green', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
plt.title("Best Random Forest: Predicted vs. Actual Values")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.grid(True)
plt.show()





#-->> Summary and Insights
#summarizing model performance
performance_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [rmse, rmse_rf],
    "MAE": [mae, mae_rf],
    "R²": [r2, r2_rf]
})

print("\nModel Performance Comparison:")
print(performance_comparison)




#time-series visualization of actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', color='blue')
plt.plot(y_pred_rf[:100], label='Predicted (RF)', color='orange')
plt.title("Actual vs. Predicted Energy Usage (Sample)")
plt.xlabel("Time Index (Sample)")
plt.ylabel("Global Active Power")
plt.legend()
plt.grid(True)
plt.show()