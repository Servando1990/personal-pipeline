

from xmlrpc.client import boolean


def fun ( string: str) -> boolean:
    if len(string) == len(list(set(string))):
        return True
    else:
        return  False

fun('abcc')
    