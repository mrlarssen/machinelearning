import copy
def rotate(arr):
    r = copy.deepcopy(arr)
    s = len(r)
    for i in range(0, s):
        for j in range(0, i+1):
            r[i][-1-j] = arr[-1-j][s-i-1]
    return r
    
A = [[int(i) for i in line.split(" ")] for line in  open("luke21.txt").readlines()]
B = rotate(A)
C = rotate(B)

def find_max(i, pyramid):
    if len(pyramid[i]) == 1:
        return max(pyramid[1]) + pyramid[i][0]
    else:
        for j in range(len(pyramid[i])):
            pyramid[i+1][j] = max([pyramid[i+1][j] + pyramid[i][j], pyramid[i+1][j+1] + pyramid[i][j]])
            
        pyramid[i] = pyramid[i+1]
        return find_max(i-1, pyramid)

import time
start = time.time()
print 'A', find_max(len(A)-2, A)
print 'B', find_max(len(B)-2, B)
print 'C', find_max(len(C)-2, C)
print (time.time() - start) * 1000, 'ms'


