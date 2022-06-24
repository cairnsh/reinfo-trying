import code
import rlcompleter
import readline

def go(l):
    readline.set_completer(rlcompleter.Completer(l).complete)
    readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(l).interact()
