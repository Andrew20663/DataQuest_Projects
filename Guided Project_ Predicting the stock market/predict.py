import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime


columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
df = pd.read_csv('sphist.csv', parse_dates = True)
df.head()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date', ascending = True)

df['Average Price 5 Days Open'] = df['Open'].rolling(window = 5, center = False).mean()
df['Average Price 5 Days High'] = df['High'].rolling(window = 5, center = False).mean()
df['Average Price 5 Days Low'] = df['Low'].rolling(window = 5, center = False).mean()
df['5 Days Volume'] = df['Volume'].rolling(window = 5, center = False).mean()
df['Year'] = df['Date'].apply(lambda x: x.year)

df['DOW'] = df['Date'].apply(lambda x: x.weekday)
dow_df = pd.get_dummies(df['DOW'])
df = pd.concat([df, dow_df], axis = 1) 
df = df.drop(['DOW'], axis = 1)

df['Average Price 5 Days Open'] = df['Average Price 5 Days Open'].shift(1)
df['Average Price 5 Days High'] = df['Average Price 5 Days High'].shift(1)
df['Average Price 5 Days Low'] = df['Average Price 5 Days Low'].shift(1)
df['5 Days Volume'] = df['5 Days Volume'].shift(1)

df = df[df['Date'] >= datetime(year = 1951, month = 1, day = 3)]
df.dropna(axis = 0) 

train = df[df['Date'] < datetime(year = 2013, month = 1, day = 1)]
test = df[df['Date'] >= datetime(year = 2013, month = 1, day = 1)]

features = ['Average Price 5 Days Open', 'Average Price 5 Days High', 'Average Price 5 Days Low', '5 Days Volume', 'Year', 0, 1, 2, 3, 4]

lr = LinearRegression()
lr.fit(train[features], train['Close'])
prediction = lr.predict(test[features])
mse = mean_squared_error(prediction, test['Close']) 
rmse = np.sqrt(mse)
                                                   

