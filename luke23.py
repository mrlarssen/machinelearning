m = {}

def f(n):
    if n in m: return m[n]

    if n == 1: return 1
    if n == 2: return 2
    if n == 3: return 4

    m[n] = f(n-1) + f(n-2) + f(n-3)

    return m[n]

print(f(250))


    