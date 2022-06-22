import torch
from vmpo import VMPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 6, {
    (0, 0): 0.03,
    (0, 1): 0.00,
    (1, 0): 0.04,
    (1, 1): 0.01,
})

class RecorderCallback:
    def __init__(self, filename):
        self.recordf = open(filename, "w")
    def special_record_games(self, opponents, actions):
        self.recordf.write("%s %s\n" % (opponents, actions))

from pseudo import pseudo
import numpy
import interactive_prompt
import sys

log = open("out.log", "w")
actual_write = sys.stdout.write
def print_and_log_to_a_file(text):
    actual_write(text)
    log.write(text)
    log.flush()
sys.stdout.write = print_and_log_to_a_file

class tit_for_tat(pseudo):
    def decide(self, opponent):
        if opponent == 2:
            # first round
            return 0
        return opponent
    def save_data(self):
        return "Tit for tat\n"

class all_cooperate(pseudo):
    def decide(self, opponent):
        return 0
    def save_data(self):
        return "All cooperate\n"

class all_defect(pseudo):
    def decide(self, opponent):
        return 1
    def save_data(self):
        return "All defect\n"

import signal
prompt = [False]
def ask_for_interactive_prompt(signum, frame):
    print("Opening a prompt at the end of the epoch.")
    prompt[0] = True
signal.signal(signal.SIGQUIT, ask_for_interactive_prompt)
print("Hit Ctrl-\ to get a prompt.")

if True:
    # Instantiate the agent
    model = SeveralAlgorithms([
        VMPO('MlpPolicy', env, batch_size=128, policy_kwargs={'optimizer_class': torch.optim.SGD, 'optimizer_kwargs': {'weight_decay': 1e-4}}, learning_rate=1e-3, gamma=0.9, verbose=1)
        for j in range(6)
    ], env)
    # Train the agent
    callback = RecorderCallback("recording.txt")
    def savemodel(model, timesteps):
        model.save("partial/vmpo-%d" % timesteps)
        if prompt[0]:
            interactive_prompt.go(locals())
            prompt[0] = False
    callback.after_training_epoch = savemodel
    model.learn(callback=callback, total_timesteps=int(400000))
    # Save the agent
    model.save("vmpo")
