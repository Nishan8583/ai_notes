import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
import math

df = pd.read_csv("house_prices.csv")

size=df["sqft_living"]
price=df["price"]

# ignore array index
x = np.array(size).reshape(-1,1)
y=np.array(price).reshape(-1,1)

#print(x,y)

model = LinearRegression()

# train the model
model.fit(x,y)
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value:", model.score(x,y))

#this is the b0
print(model.coef_[0])  # slope
#this is b1 in our model
print(model.intercept_[0])

#plot our size and price data set as green
plt.scatter(x, y, color= 'green')
# predict, the old x and new x, and plot the line
plt.plot(x, model.predict(x), color = 'black')
plt.title ("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()
