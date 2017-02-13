bank = dict()
for sender,receiver,amount in [l.strip().split(',') for l in open('transaksjoner.txt').readlines()]:
    if sender != 'None': bank[sender] -= int(amount)
    bank[receiver] = bank.get(receiver,0) + int(amount)

print len([v for v in bank.values() if v > 10])