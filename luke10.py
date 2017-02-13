"""
Trollmann, Kriger, Prest, Tyv

17 mennesker fanget

100 rom, rom nr = antall goblins


1. Hvis tyv er i live => dreper 1 goblin
Hvis trollmann er i live => dreper max 10 goblin
Hvis krigeren er i live => 1 goblin

Hvis presten er i live og en annen ikke, først gjennopplive kriger så trollmann. Maks 1 per rom.

Hvis tyv eneste som er i live => går til neste rom

Hvis 10x flere goblins enn eventyrere. Dreper først krigeren, trollmann, prest.

Hvis fremdeles eventyrere og goblins i rommet, gå til punkt 1 (ny runde i samme rom), hvis ikke, gå til neste rom

"""

rooms = range(1,101)

adventurers = {
    'tyv': {'is_alive': True, 'died_at': 0},
    'trollmann': {'is_alive': True, 'died_at': 0},
    'kriger': {'is_alive': True, 'died_at': 0},
    'prest': {'is_alive': True, 'has_revived': False, 'died_at': 0},
}

def n_adventurers_alive():
    n_alive = 0
    if adventurers['tyv']['is_alive']:
        n_alive += 1
    if adventurers['trollmann']['is_alive']:
        n_alive += 1
    if adventurers['kriger']['is_alive']:
        n_alive += 1
    if adventurers['prest']['is_alive']:
        n_alive += 1

    return n_alive

survived = 0
for room in rooms:
    goblins = room
    adventurers['prest']['has_revived'] = False
    while goblins > 0:
        if adventurers['tyv']['is_alive'] and goblins > 0:
            goblins -= 1
        if adventurers['trollmann']['is_alive'] and goblins > 0:
            if goblins - 10 >= 0:
                goblins -= 10
            else:
                goblins = 0
        if adventurers['kriger']['is_alive'] and goblins > 0:
            goblins -= 1
        if adventurers['prest']['is_alive']:
            if (not adventurers['kriger']['is_alive']) and (not adventurers['prest']['has_revived']) and adventurers['kriger']['died_at'] == room:
                adventurers['kriger']['is_alive'] = True
                adventurers['prest']['has_revived'] = True
            elif (not adventurers['trollmann']['is_alive']) and (not adventurers['prest']['has_revived']) and adventurers['trollmann']['died_at'] == room:
                adventurers['trollmann']['is_alive'] = True
                adventurers['prest']['has_revived'] = True
        if adventurers['tyv']['is_alive'] and (not adventurers['trollmann']['is_alive']) and (not adventurers['kriger']['is_alive']) and (not adventurers['prest']['is_alive']):
            survived += goblins
            goblins = 0
        elif n_adventurers_alive()*10 <= goblins:
            if adventurers['kriger']['is_alive']:
                adventurers['kriger']['is_alive'] = False
                adventurers['kriger']['died_at'] = room
            elif adventurers['trollmann']['is_alive']:
                adventurers['trollmann']['is_alive'] = False
                adventurers['trollmann']['died_at'] = room
            elif adventurers['prest']['is_alive']:
                adventurers['prest']['is_alive'] = False
                adventurers['prest']['died_at'] = room

print adventurers
print n_adventurers_alive()
print survived
print n_adventurers_alive() + survived + 17


