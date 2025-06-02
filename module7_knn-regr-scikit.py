import numpy as np
from sklearn.neighbors import KNeighborsRegressor
class KNNRegressor:
    def __init__(self):
        self.points = []  # Store as list of tuples (x, y)

    def insert_point(self, x, y):
        self.points.append((x, y))

    def get_variance(self):
        if not self.points:
            return 0.0
        y_vals = np.array([p[1] for p in self.points])
        return np.var(y_vals)

    def compute(self, target_x, k):
        if k > len(self.points):
            raise ValueError(f"k: {k} is greater than number of points N: {len(self.points)}")

        data = np.array(self.points)
        X = data[:, 0].reshape(-1, 1)
        y = data[:, 1]

        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X, y)
        y_pred = model.predict(np.array([[target_x]]))
        return y_pred[0]
    
knn = KNNRegressor()
print("Enter the number of points (N):")
N =int(input())
print("Enter the number of neighbors (k):")
k = int(input())

for i in range(N):
    print(f"Enter x value for point #{i+1}:")
    x = float(input())
    print(f"Enter y value for point #{i+1}:")
    y = float(input())
    knn.insert_point(x, y)

print("Enter the value of X to predict Y:")
target_x = float(input())

try:
    prediction = knn.compute(target_x, k)
    variance = knn.get_variance()
    print(f"Predicted Y using {k}-NN regression: {prediction}")
    print(f"Variance of training labels: {variance}")

except ValueError as e:
    print("Error:", e)