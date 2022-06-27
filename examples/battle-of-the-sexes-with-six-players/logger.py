import sys

def redirect_stdout_to_a_log_file(fnam):
    log = open(fnam, "w")
    actual_write = sys.stdout.write
    def print_and_log_to_a_file(text):
        actual_write(text)
        log.write(text)
        log.flush()
    sys.stdout.write = print_and_log_to_a_file
