"""Keep loader chatter in logs; reserve the terminal for case/epoch/batch rows."""
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import sys
import traceback
from tranad_plus.campaigns.exploration.progress import enabled


class Stream:
    def __init__(self, console, logfile):
        self.console, self.logfile, self.pending = console, logfile, ''
    def write(self, text):
        self.logfile.write(text)
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if enabled() and line.startswith(('[DATA]', 'GPU CACHE', '[TRAIN]')):
                continue
            if enabled():
                from tqdm import tqdm
                tqdm.write(line, file=self.console)
            else:
                self.console.write(line+'\n')
        return len(text)
    def flush(self):
        self.logfile.flush(); self.console.flush()
    def isatty(self):
        return False


@contextmanager
def logged(path):
    with Path(path).open('a', buffering=1) as stream:
        with redirect_stdout(Stream(sys.stdout, stream)):
            try:
                yield
            except Exception:
                traceback.print_exc(file=stream)
                raise
