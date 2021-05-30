def isfloat(x):
    if type(x) == float:
        return True
    return False

def isint(x):
    if type(x) == int:
        return True
    return False

def isdigit(x):
    return isint(x) or isfloat(x)
