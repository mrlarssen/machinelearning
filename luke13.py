instructions = [l.strip().split() for l in open('instruksjoner.txt').readlines()]
grid = [[0 for col in xrange(10000)] for row in xrange(10000)]
#instructions = [["turn" ,"on" ,"0,0" ,"through" ,"1,1"]]
for ins in instructions:
    print ins
    if ins[0] == 'turn':
        val = 0
        if ins[1] == 'on':
            val = 1
        x_from,y_from = ins[2].split(',')
        x_to,y_to = ins[4].split(',')
        i = int(x_from)
        j = int(y_from)
        while i <= int(x_to):
            while j <= int(y_to):
                grid[i][j] = val
                j += 1
            j = int(y_from)
            i += 1
    else:
        x_from,y_from = ins[1].split(',')
        x_to,y_to = ins[3].split(',')
        i = int(x_from)
        j = int(y_from)
        while i <= int(x_to):
            while j <= int(y_to):
                if grid[i][j] == 1:
                    grid[i][j] = 0
                else:
                    grid[i][j] = 1
                j += 1
            j = int(y_from)
            i += 1
                    
print len(filter(lambda x: x == 1, [j for i in grid for j in i]))
    