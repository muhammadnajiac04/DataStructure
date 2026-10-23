from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

iris = load_iris()
X, y = iris.data, iris.target

a, b, c, d = train_test_split(X, y, test_size=0.3)

m = LinearRegression()
m.fit(a, c)
p = m.predict(b)

print(p)
print("R2 score:", r2_score(d, p))
print("MSE:", mean_squared_error(d, p))