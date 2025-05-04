array = []
print("Please enter number of numbers: N")
n = int(input())

for i in range(n):
  print("Please enter number index " + str(i))
  array.append(int(input()))

print("Please enter number X")
x = int(input())
try:
  print("Found at index: " + str(array.index(x)))
except:
  print("Not Found")
  print(-1)