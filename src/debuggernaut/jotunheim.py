'''
Laufeyspawn: 
 - Track and manage Frost Giant 'invasions' (debug traces) during runtime.

Inspired by Jotunheim's chaotic threats to Asgard, this module marks the presence
of bugs and enables tracing their movement through the system.

- laufeyspawn: 
    Decorator to mark and trace function invasions.
- jotunbane: 
    Utility to check if current function is in debug-traced state.
'''

from typing import Callable, TypeVar, Any
import threading
from .bifrost import RD, BU, YW
import functools

# thread-local storage to track nested debug states
RIME = _debug_stack = threading.local()
F = TypeVar('F', bound = Callable)

class laufeyspawn:
    '''
    A decorator class representing an invasion of bugs (laufeyspawn).

    When summoned, prints debug information on function call entry and exit.
    Uses thread-local tracking to correctly manage nested decorated functions.

    Parameters:
    -----------
    summoned : bool
        Whether debug tracing is active for the decorated function.
    '''
    def __init__(self, *, summoned: bool = False):
        '''
        Initialize the laufeyspawn decorator.

        Parameters:
        -----------
        summoned : bool
            Set True to enable debug tracing for decorated functions.
        '''
        self.summoned = summoned

    def __call__(self, func: F) -> F:
        '''
        Wrap the target function to print debug information if summoned.

        Parameters:
        -----------
        func : Callable
            The function to decorate.

        Returns:
        --------
        Callable
            The wrapped function.
        '''
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # create thread-local stack if it doesn't exist yet
            if not hasattr(_debug_stack, 'nest'):
                _debug_stack.nest = []

            # push current 'summoned' state onto the stack
            _debug_stack.nest.append(self.summoned)

            # try to get the class name if this is a method
            class_name = None
            if args:
                class_name = args[0].__class__.__name__

            # print entry debug statement if summoned
            if self.summoned:
                if class_name:
                    print(f'{RD("[DEBUG]")} {BU("Calling")} \'{class_name}.{func.__name__}\'')
                else:
                    print(f'{RD("[DEBUG]")} {BU("Calling")} \'{func.__name__}\'')

            try:
                # call the actual function
                return func(*args, **kwargs)
            finally:
                # --- pop the 'summoned state' --- 
                _debug_stack.nest.pop()

                # --- exit debug statement ---
                if self.summoned:
                    if class_name:
                        print(f'{RD("[DEBUG]")} {YW("Exiting")} \'{class_name}.{func.__name__}\'')
                    else:
                        print(f'{RD("[DEBUG]")} {YW("Exiting")} \'{func.__name__}\'')

        return wrapper

    @staticmethod
    def jotunbane() -> bool:
        '''
        Check if the current function call is under debug tracing.

        Returns:
        --------
        bool
            True if the current function is inside a debug-traced call stack.
        '''
        return hasattr(RIME, 'nest') and RIME.nest and RIME.nest[-1]

jotunbane = laufeyspawn.jotunbane
__all__ = ['laufeyspawn', 'jotunbane']
