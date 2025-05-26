import numpy as np

class KNNRegressor:
    def __init__(self):
        self.data = np.empty((0, 2))  # shape: (N, 2)

    def insert_point(self, x, y):
        point = np.array([[x, y]])
        self.data = np.vstack((self.data, point))  # Efficient batch-wise NumPy appending

    def compute(self, target_x, k):
        N = self.data.shape[0]
        if k > N:
            raise ValueError(f"k = {k} is greater than number of points N = {N}")

        x_vals = self.data[:, 0]
        y_vals = self.data[:, 1]

        # Compute distances using vectorized NumPy operations
        distances = np.abs(x_vals - target_x)

        # Get indices of k smallest distances
        nearest_indices = np.argpartition(distances, k)[:k]

        # Compute mean y
        nearest_y = y_vals[nearest_indices]
        return np.mean(nearest_y)
    
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
    print(f"Predicted Y using {k}-NN regression: {prediction}")
except ValueError as e:
    print("Error:", e)