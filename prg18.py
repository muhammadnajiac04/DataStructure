from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = load_diabetes()
X = data.data[:, [2]]
y = data.target

a, b, c, d = train_test_split(X, y, test_size=0.3)

m = LinearRegression()
m.fit(a, c)
p = m.predict(b)

print("R2 score:", r2_score(d, p))
print("MSE:", mean_squared_error(d, p))

plt.scatter(b, d, label="Actual")
plt.plot(b, p, color="red", label="Predicted")
plt.title("Linear Regression")
plt.legend()
plt.show()