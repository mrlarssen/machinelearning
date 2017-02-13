i = 0
numbers = []

for n in range(1, 1338):
    if "7" in str(n) or n % 7 == 0:
        numbers.append(numbers[i])
        i += 1
    else:
        numbers.append(n)

print numbers[-1]