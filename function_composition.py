
import functools
from typing import Callable

ComposableFunction = Callable[[float], float] # input [float] output float

def compose(*functions: ComposableFunction) -> ComposableFunction:
    """ helper functions that allows to iterate over intermidiate results of any functions
        reduce: call through the list and keeps track of intermitdate result and return final result
        f, g: functions

    Returns:
        ComposableFunction: function
    """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def add_Three(x: float) -> float:
    return + 3

def multiplyByTwo(x: float) -> float:
    return x * 2

def main():
    x = 12
    myfunc = compose(add_Three, add_Three, multiplyByTwo)
    result = myfunc(x)
    # x = add_Three(x)
    # x = multiplyByTwo(x)
    print(f'Result: {result}')

if __name__ == '__main__':
    main()
