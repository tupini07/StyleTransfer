import colored_traceback
from stransfer.clis import cli

if __name__ == "__main__":
    colored_traceback.add_hook()
    cli(**{}, prog_name='stransfer')  # suppress warning
