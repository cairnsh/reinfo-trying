import re

PLAYERS = 6

rec = open("recording.txt")

DIGITS = re.compile(r'\d+')
class memory:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, z):
        self.sum += z
        self.count += 1

    def get_average_and_clear(self):
        try:
            return self.sum/self.count
        except ZeroDivisionError:
            return None
        finally:
            self.sum = 0
            self.count = 0
            pass
        
history = {(i, j): memory() for i in range(PLAYERS) for j in range(PLAYERS) if i != j}

reward = {
    (0, 0): 0,
    (0, 1): 0.5,
    (1, 0): 1,
    (1, 1): 0
}

def say():
    for a in range(PLAYERS):
        out = [None] * PLAYERS
        for b in range(PLAYERS):
            if a == b:
                what = ""
            else:
                average = history[a,b].get_average_and_clear()
                if average is None:
                    what = "None"
                else:
                    what = "%.2f"%average
            out[b] = what
        print((" " + " %9s" * PLAYERS) % tuple(out))
    print()

count = 0
for line in rec:
    matches = [int(z) for z in DIGITS.findall(line)]
    perm = matches[:PLAYERS]
    ac = matches[PLAYERS:]
    for j in range(PLAYERS):
        a = j
        b = perm[j]
        history[a,b].add(reward[ac[j], ac[perm[j]]])

    count += 1
    if count == 7000:
        say()
        count = 0
