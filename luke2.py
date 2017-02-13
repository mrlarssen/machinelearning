

def sumEven(arr):
    sumTot = 0
    for n in arr:
        if n % 2 == 0:
          sumTot += n
          
    print(sumTot)
    return sumTot

def fib():
    nums = [1,1]
    while sumEven(nums) < 4000000000:
        nextFib = nums[-1] + nums[-2]
        nums.append(nextFib)
        
fib()
        