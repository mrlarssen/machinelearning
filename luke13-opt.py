import numpy as np
import re

size = 10000
leds = np.zeros((size, size), dtype=bool)

for line in open("instruksjoner.txt", "r").readlines():
    match = re.match(r"(?P<instruksjoner>.*) (?P<from>\d+,\d+) through (?P<to>\d+,\d+)", line)
    command = match.group("instruksjoner")
    f_x, f_y = [int(f) for f in match.group("from").split(",")]
    t_x, t_y = [int(t)+1 for t in match.group("to").split(",")]
    if command == "turn on":
        leds[f_x:t_x, f_y:t_y] = True
    elif command == "toggle":
        leds[f_x:t_x, f_y:t_y] ^= True
    elif command == "turn off":
        leds[f_x:t_x, f_y:t_y] = False

print(leds.sum())