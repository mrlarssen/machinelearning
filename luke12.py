print [{ 0: "0", 1: "I", 2: "II",3: "III",4: "IV",5: "V",6: "VI",7: "VII",8: "VIII",9: "IX",10: "X",11: "XI",12: "XII",13: "XIII"}[int(((ord(m)-96)/2.0)+0.5)] for m in filter(lambda x: x not in "?!., ", "Your message was received with gratitude! We do not know about you, but Christmas is definitely our favourite holiday. The tree, the lights, all the presents to unwrap. Could there be anything more magical than that?! We wish you a happy holiday and a happy new year!".lower())] + [{ 0: "0", 1: "I", 2: "II",3: "III",4: "IV",5: "V",6: "VI",7: "VII",8: "VIII",9: "IX",10: "X",11: "XI",12: "XII",13: "XIII"}[(ord(m)-96)/2] for m in filter(lambda x: x not in "?!., ", "Your message was received with gratitude! We do not know about you, but Christmas is definitely our favourite holiday. The tree, the lights, all the presents to unwrap. Could there be anything more magical than that?! We wish you a happy holiday and a happy new year!".lower())][::-1]