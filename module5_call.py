from module5_mod import NumberManager


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