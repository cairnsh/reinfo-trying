import torch
from vmpo import VMPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

from logger import redirect_stdout_to_a_log_file
from pseudo import tit_for_tat, all_cooperate, all_defect
from interactive_prompt import InteractivePromptHandler

env = MatrixGameVecEnv(2, 6, {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (1, 1): 0,
})

class RecorderCallback:
    def __init__(self, filename):
        self.recordf = open(filename, "w")
    def special_record_games(self, actions):
        self.recordf.write("%s\n" % actions)

redirect_stdout_to_a_log_file("out.log")

prompt = InteractivePromptHandler("Opening a prompt at the end of the epoch.")

vmpo_options = dict(
    batch_size=128,
    policy_kwargs = dict(
        #optimizer_class=torch.optim.SGD,
    ),
    learning_rate=1e-3,
    gamma=0.99,
    verbose=1
)

if True:
    # Instantiate the agent
    model = SeveralAlgorithms(
        [VMPO('MlpPolicy',env,**vmpo_options)for _ in range(6)],
    env)
    # Train the agent
    callback = RecorderCallback("recording.txt")
    def savemodel(model, timesteps):
        model.save("partial/vmpo-%d" % timesteps)
        prompt.start_if_requested(lo=locals())

    callback.after_training_epoch = savemodel
    model.learn(callback=callback, total_timesteps=int(1000000))
    # Save the agent
    model.save("vmpo")
