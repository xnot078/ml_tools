def chain(start, *funcs):
    res = start
    for f in funcs:
        res = f(res)
    return res

if __name__ == '__main__':
    x1 = lambda x: x+1
    x2 = lambda x: x**2
    print(chain(1, x1, x2))
