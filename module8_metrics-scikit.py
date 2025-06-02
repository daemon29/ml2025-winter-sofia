import numpy as np
from sklearn.metrics import precision_score, recall_score

class BinaryClassifier:
    def __init__(self, n):
        self.y_true = np.empty(n, dtype=np.int8)
        self.y_pred = np.empty(n, dtype=np.int8)
        self.index = 0
        self.n = n

    def insert_point(self, ground_truth, predicted):
        if self.index >= self.n:
            return ValueError("Too many value, must be equal")
        self.y_true[self.index] = ground_truth
        self.y_pred[self.index] = predicted
        self.index += 1

    def compute_metrics(self):
        if self.index == 0:
            raise ValueError("No data to compute metrics.")
        precision = precision_score(self.y_true[:self.index], self.y_pred[:self.index], zero_division=0)
        recall = recall_score(self.y_true[:self.index], self.y_pred[:self.index], zero_division=0)
        return precision, recall

def main():

    print("Enter the number of points (N):")
    N = int(input())
    evaluator = BinaryClassifier(N)

    for i in range(N):
        print(f"Enter groundtruth label for #{i+1}:")
        x = int(input())
        print(f"Enter predicted label for #{i+1}:")
        y = int(input())
        evaluator.insert_point(x, y)

    
    precision, recall = evaluator.compute_metrics()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
  

if __name__ == "__main__":
    main()