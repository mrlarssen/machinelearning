"""
Lovlige trekk
    (0,0) => (2,1)
    (0,0) => (1,2)
    (0,0) => (-2,1)
    (0,0) => (-1,2)
    (0,0) => (-2,-1)
    (0,0) => (-1,-2)
    (0,0) => (2,-1)
    (0,0) => (1,-2)
    
    (x,y) => (x+2, y+1)
    (x,y) => (x+1, y+2)
    (x,y) => (x-2, y+1)
    (x,y) => (x-1, y+2)
    (x,y) => (x-2, y-1)
    (x,y) => (x-1, y-2)
    (x,y) => (x-2, y-1)

"""

legal_moves = [
    (2,1,3),
    (1,2,3),
    (-2,1,1), 
    (-1,2,1),
    (-2,-1,3),
    (-1,-2,3), 
    (2,-1,1),
    (1,-2,1)
    ]

start = (0,0)
visited = []
max_distance = 0

def move():
    pos = (0,0,0)
    count = 0
    max_distance = 0
    while count < 1000000:
        pos,distance = find_next_move(pos[0], pos[1], pos[2])
        if distance > max_distance:
            max_distance = distance
        count += 1
    
    print max_distance
        
        
def find_next_move(x,y,value):
    
    possible_moves = [(x+t[0], y+t[1], t[2]) for t in legal_moves]
    possible_values = [t[2] for t in possible_moves]
    
    minv = min(map(lambda x: abs(x-value), possible_values))
    min_v_indexes = [i for i,v in enumerate(possible_values) if abs(v-value) == minv]

    min_tuple = (0,0)
    
    if len(min_v_indexes) > 0:
        min_xs = find_min_x([possible_moves[i] for i in min_v_indexes])
        if len(min_xs) > 0:
            min_tuple = find_min_y(min_xs)
        else:
            min_tuple = min_xs[0]
    else:
        min_tuple = possible_moves[min_v_indexes[0]]
    
    if min_tuple in visited:
        val = 0
    else:
        val = 1000
    
    final = (min_tuple[0], min_tuple[1], val)
   
    return final, abs(value + minv)
        
def find_min_x(l):
    minx = min([x for (x,y,z) in l])
    return [v for v in l if v[0] == minx]
    
def find_min_y(l):
    miny = min([y for (x,y,z) in l])
    return [v for v in l if v[1] == miny][0]

move()


