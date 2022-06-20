import torch
from vmpo import VMPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 4, {
    (0, 0): 0.03,
    (0, 1): 0.00,
    (1, 0): 0.04,
    (1, 1): 0.01
})

class RecorderCallback:
    def __init__(self, filename):
        self.recordf = open(filename, "w")
    def special_record_games(self, opponents, actions):
        self.recordf.write("%s %s\n" % (opponents, actions))

from pseudo import pseudo
import numpy

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

if True:
    # Instantiate the agent
    model = SeveralAlgorithms([tit_for_tat(env), all_cooperate(env), all_defect(env)] + [
        VMPO('MlpPolicy', env, batch_size=128, learning_rate=1e-3, gamma=0.9, verbose=1)
        for j in range(1)
    ], env)
    # Train the agent
    import interactive_prompt
    interactive_prompt.go(locals())
    callback = RecorderCallback("recording.txt")
    def savemodel(model, timesteps):
        model.save("prisoners_dilemma-%d" % timesteps)
    callback.after_training_epoch = savemodel
    model.learn(callback=callback, total_timesteps=int(100000))
    # Save the agent
    model.save("prisoners_dilemma")
