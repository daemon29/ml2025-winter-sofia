import numpy as np

class KNNRegressor:
    def __init__(self):
        self.points = []  # List of points

    def insert_point(self, x, y):
        self.points.append((x, y))

    def compute(self, target_x, k):
        if k > len(self.points):
            raise ValueError(f"k = {k} is greater than number of points N = {len(self.points)}")

        # Convert list of tuples into a numpy array
        data = np.array(self.points)  # shape: (N, 2)
        x_vals = data[:, 0]
        y_vals = data[:, 1]

        # Compute distances from each x to target_x
        distances = np.abs(x_vals - target_x)

        # Get indices of k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Compute mean y of those k neighbors
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