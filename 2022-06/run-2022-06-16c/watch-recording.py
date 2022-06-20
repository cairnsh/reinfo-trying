import re

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
        
history = {(i, j): memory() for i in range(4) for j in range(4) if i != j}

reward = {
    (0, 0): 3,
    (0, 1): 0,
    (1, 0): 4,
    (1, 1): 1
}

def say():
    for a in range(4):
        out = [None] * 4
        for b in range(4):
            if a == b:
                what = ""
            else:
                average = history[a,b].get_average_and_clear()
                if average is None:
                    what = "None"
                else:
                    what = "%.2f"%average
            out[b] = what
        print("  %9s %9s %9s %9s" % tuple(out))
    print()

count = 0
for line in rec:
    matches = [int(z) for z in DIGITS.findall(line)]
    perm = matches[:4]
    ac = matches[4:]
    for j in range(4):
        a = j
        b = perm[j]
        history[a,b].add(reward[ac[j], ac[perm[j]]])

    count += 1
    if count == 1000:
        say()
        count = 0
