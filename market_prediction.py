import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


ticker = "SPY"
data = yf.download(ticker, start="2010-01-01", end="2024-01-01")
data = data[["Close"]]

data["returns"] = data["Close"].pct_change()
data["ma5"] = data["Close"].rolling(5).mean()
data["ma20"] = data["Close"].rolling(20).mean()
data["volatility"] = data["returns"].rolling(20).std()
data["momentum"] = data["Close"] / data["Close"].shift(10)

data["target"] = (data["returns"].shift(-1) > 0).astype(int)
data = data.dropna()

features = ["ma5", "ma20", "volatility", "momentum"]
X = data[features]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model accuracy:", accuracy)

strategy_returns = data["returns"].iloc[-len(predictions):] * predictions
cumulative_strategy = (1 + strategy_returns).cumprod()
cumulative_market = (1 + data["returns"].iloc[-len(predictions):]).cumprod()

plt.figure(figsize=(10,6))
plt.plot(cumulative_market, label="Market")
plt.plot(cumulative_strategy, label="ML Strategy")
plt.legend()
plt.title("Machine Learning Trading Strategy vs Market")
plt.show()
