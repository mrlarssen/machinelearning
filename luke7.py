summary = dict.fromkeys(['north','east','south','west'],0)
for line in [l.strip().split() for l in open('skattekart.txt').readlines()]:
    summary[line[3]] += int(line[1])
    
print(summary['north']-summary['south'], summary['west']-summary['east'])
    
    
