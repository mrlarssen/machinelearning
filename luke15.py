
number = '1111321112321111'


    
def look_and_say(number):
    i = 0
    res = ''
    while i < len(number):
        index = 1
        count = 1
        while (i+index < len(number)) and number[i] == number[i+index]:
            index += 1
            count += 1
        
        res += (str(count) + number[i])
        i += index

    return res

#number = '132113'
for _ in range(50):
    number = look_and_say(number)
    
print len(number)