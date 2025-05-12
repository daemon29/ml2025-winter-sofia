class NumberManager:
    def __init__(self):
        self.numbers = []

    def insert_number(self, number):
        self.numbers.append(number)

    def find_number(self, x):
        try:
            return self.numbers.index(x) + 1
        except ValueError:
            return -1

def main():
    manager = NumberManager()

    print("Please enter number of numbers: N")
    n = int(input())

    for i in range(n):
        print(f"Please enter number index {i}")
        number = int(input())
        manager.insert_number(number)

    print("Please enter number X")
    x = int(input())

    result = manager.find_number(x)
    if result == -1:
        print("Not Found")
        print(-1)
    else:
        print(f"Found at index: {result}")

if __name__ == "__main__":
    main()