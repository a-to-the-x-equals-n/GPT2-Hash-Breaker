from .bifrost import GR, PR, YW, RD
import shutil
from pathlib import Path
import inspect

class Heimdahl:
    @staticmethod
    def __call__(s: str, unveil: bool = False, threat: int = 0) -> None:
        if not unveil:
            return

        match threat:
            case 0: CLR = PR 
            case 1: CLR = GR
            case 2: CLR = YW
            case 3: CLR = RD
            case _: CLR = GR

        try:
            w = shutil.get_terminal_size().columns
        except Exception:
            w = 40
        s = f' {s} ({Path(inspect.stack()[1].filename).name}) '
        pad = (w - len(s)) // 2
        line = '=' * pad + s + '=' * pad
        line = line.ljust(w, '=')
        print(CLR(line))

heimdahl = Heimdahl()
__all__ = ['heimdahl']


if __name__ == '__main__':
    print('\n[TEST] Heimdahl threat level output:\n')
    for level in range(4):
        heimdahl(f'THREAT LEVEL {level}', unveil = True, threat = level)
    print('\n[TEST] Heimdahl with unveil = False (should NOT print):\n')
    heimdahl('YOU SHOULD NOT SEE THIS', unveil = False, threat = 3)
    print('\n[TEST COMPLETE]')
