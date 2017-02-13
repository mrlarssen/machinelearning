from itertools import combinations as comb
from math import factorial
import time

t0 = time.time()

print min([(sum(map(lambda x: factorial(x), i)), i) for l in [filter(lambda x: reduce(lambda x, y: x*y, map(lambda x: x**x, list(x))) > 2.25e32, [a for b in range(2,17,2) for a in comb(range(1,17), b)])] for i in l], key=lambda t: t[0])[1]

print time.time() - t0