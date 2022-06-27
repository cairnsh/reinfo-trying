import re

LINE_REGULAR_EXPRESSION = re.compile(
    r'\[(\d+(?:\s+\d+)*)\]' +
    r'\s+' +
    r'\[(\d+(?:,\s+\d+)*)\]' +
    r'\s+'
)

DIGITS = re.compile(r'\d+')

def get_all_digit_strings(z):
    return [int(it) for it in DIGITS.findall(z)]

def parse_recording():
    fo = open("recording.txt")
    nplayer = None
    for line in fo:
        m = LINE_REGULAR_EXPRESSION.match(line)
        permutation = get_all_digit_strings(m[1])
        actions = get_all_digit_strings(m[2])
        
        if nplayer is None:
            nplayer = len(permutation)
        else:
            assert nplayer == len(permutation)
        
        assert nplayer == len(actions)
        
        yield permutation, actions

def load_the_recording():
    recording = list(parse_recording())
    print("Loaded recording: %d games" % len(recording))
    return len(recording[0][0]), recording

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

def calculate_reward_tables(reward, nplayers, recording):
    history = {
        (i, j): memory() for i in range(nplayers) for j in range(nplayers)
            if i != j
    }
    
    count = 0
    for entry in recording:
        permutation, action = entry
        for j in range(nplayers):
            a = j
            b = permutation[j]
            history[a, b].add(reward[action[a], action[b]])
        count += 1
        if count % 8000 == 0:
            yield {
                (i, j): history[i, j].get_average_and_clear()
                for i in range(nplayers) for j in range(nplayers) if i != j
            }

def calculate_rewards_per_agent(reward, nplayers, recording):
    history = [memory() for _ in range(nplayers)]
    
    count = 0
    for entry in recording:
        permutation, action = entry
        for j in range(nplayers):
            a = j
            b = permutation[j]
            history[a].add(reward[action[a], action[b]])
        count += 1
        if count % 8000 == 0:
            yield [a_memory.get_average_and_clear() for a_memory in history]
            