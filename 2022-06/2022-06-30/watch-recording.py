import re

PLAYERS = 6

rec = open("recording.txt")

DIGITS = re.compile(r'\d+')

EPISODE_LENGTH = 100

def get_matching():
    matchinglog = open("matching.log")
    for line in matchinglog:
        matches = [int(z) for z in DIGITS.findall(line)]
        for _ in range(EPISODE_LENGTH):
            yield matches

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
    (0, 1): 1,
    (1, 0): 2,
    (1, 1): 0
}


#reward = {
#    (i, j): ([0, 1, -1])[(j - i) % 3]
#    for j in range(3) for i in range(3)
#    }

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
problem = memory()
matching_generator = get_matching()
for line in rec:
    perm = next(matching_generator)
    ac = [int(z) for z in DIGITS.findall(line)]
    for j in range(PLAYERS):
        a = j
        b = perm[j]
        history[a,b].add(reward[ac[j], ac[perm[j]]])
        problem.add(1 if ac[j] == ac[perm[j]] else 0)

    count += 1
    if count % 8000 == 0:
        print("%7d steps" % count)
        say()
        #print("%7d percentage matching: %.2f%%" % (count, 100*problem.get_average_and_clear()))
