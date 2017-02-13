from itertools import permutations as perms

values = ['0','1','2','3','4','5','6','7','8','9']

A = B = list(perms(values, 5))

highest = product = 0

for a in A:
    for b in B:
        if set(a).isdisjoint(set(b)):
            product = int(''.join(a)) * int(''.join(b))
            if product > highest:
                highest = product
                print highest

