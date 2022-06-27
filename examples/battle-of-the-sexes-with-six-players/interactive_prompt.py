import code
import rlcompleter
import readline
import signal

def go(l):
    readline.set_completer(rlcompleter.Completer(l).complete)
    readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(l).interact()

class InteractivePromptHandler:
    def __init__(self, it):
        self.request = False
        def ask_for_interactive_prompt(signum, frame):
            print(it)
            self.request = True
        signal.signal(signal.SIGQUIT, ask_for_interactive_prompt)
        print("Hit Ctrl-\ to get a prompt.")

    def start_if_requested(self, lo):
        if self.request:
            go(lo)
            self.request = False
