"""import string
letters = {key: value for (key, value) in zip(range(1, 27), string.ascii_uppercase)}

def conv(val, cols):
    if val > 0:
        if val < 27:
            return cols + letters[val]
        else:
            count = 1
            while 26**(count+1) <= val:
                count += 1
            i = 1
            while (26**count) * (i+1)  <= val:
                i += 1
            return conv(val - (26**count) * i, cols + letters[i])
       
print conv(90101894, '') """
from numpy import base_repr
num = 90101894
print base_repr(num, 27)



