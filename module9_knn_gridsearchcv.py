import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
class GridSearch:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.train_size = 0
        self.test_size = 0
        self.train_index = 0
        self.test_index = 0

    def init_train(self, n):
        self.x_train = np.empty(n, dtype=np.float32)
        self.y_train = np.empty(n, dtype=np.int32)
        self.train_size = n
        self.train_index = 0

    def init_test(self, m):
        self.x_test = np.empty(m, dtype=np.float32)
        self.y_test = np.empty(m, dtype=np.int32)
        self.test_size = m
        self.test_index = 0

    def insert_train(self, x, y):
        self.x_train[self.train_index] = x
        self.y_train[self.train_index] = y
        self.train_index += 1

    def insert_test(self, x, y):
        self.x_test[self.test_index] = x
        self.y_test[self.test_index] = y
        self.test_index += 1
    def search_best_k(self, max_k=10):
        X_train = self.x_train.reshape(-1, 1)
        y_train = self.y_train
        X_test = self.x_test.reshape(-1, 1)
        y_test = self.y_test

        best_k = 1
        best_acc = 0.0

        for k in range(1, min(max_k, len(X_train)) + 1):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            if acc > best_acc:
                best_acc = acc
                best_k = k

        return best_k, best_acc


def main():
    gridsearch = GridSearch()

    print("Enter the number of training samples (N):")
    N = int(input())
    gridsearch.init_train(N)

    for i in range(N):
        print(f"Enter train feature X for #{i+1}:")
        x = float(input())
        print(f"Enter train label Y for #{i+1}:")
        y = int(input())
        gridsearch.insert_train(x, y)

    print("Enter the number of testing samples (M):")
    M = int(input())
    gridsearch.init_test(M)

    for i in range(M):
        print(f"Enter test feature X for #{i+1}:")
        x = float(input())
        print(f"Enter test label Y for #{i+1}:")
        y = int(input())
        gridsearch.insert_test(x, y)
    
    # Search best k
    best_k, best_acc = gridsearch.search_best_k()
    print(f"Best k: {best_k}")
    print(f"Test accuracy: {best_acc:.2f}")
  

if __name__ == "__main__":
    main()