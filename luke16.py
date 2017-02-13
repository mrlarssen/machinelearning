from math import *

for i in range(100000,500000):
    the_sum = sum([int(y) for x in str(i) for y in x])
    if the_sum == 43:
        
        if i == int(sqrt(i)) ** 2:
            print i
            
            