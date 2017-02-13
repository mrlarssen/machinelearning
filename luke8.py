kast = [int(l.strip().split()[0]) for l in open('kast.txt').readlines()]
stiger = [(3,17), (8,10), (15,44), (22,5), (39,56), (49,75), (62,45), (64,19), (65,73), (80,12), (87,79)]
posisjoner = dict()
stige = 0
for i,k in enumerate(kast):
    spiller = (i % 1337) + 1
    old_pos, new_pos = posisjoner.get(spiller,1), posisjoner.get(spiller,1) + k
    stige_treff = [y for (x,y) in stiger if x == new_pos]
    if new_pos > 90:
        posisjoner[spiller] = old_pos
    elif new_pos == 90:
        print (spiller*stige)
        break
    elif len(stige_treff) > 0:
            stige += 1
            posisjoner[spiller] = stige_treff[0]
    else:
        posisjoner[spiller] = new_pos
