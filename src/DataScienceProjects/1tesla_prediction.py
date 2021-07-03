
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from fbprophet import Prophet

data = pd.read_csv("dataset/TSLA.csv")
print(data.head())

close = data["Close"]
ax = close.plot(title = "Tesla")
ax.set_xlabel("date")
ax.set_ylabel("close")
plt.show()


# create a new dataframe with only two columns
data["Data"] = pd.to_datetime(data["Date"], infer_datetime_format= True)
data = data[["Date","Close"]]

data = data.rename(columns = {"Date": "ds", "Close": "y"})

#Import Gaussian Naive Bayes model

