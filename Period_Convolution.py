import numpy as np

xn = [1, 2, 3, 4]
hn = [1, 1, 1, 1]


def pcon(xn, hn, n=0):
    if not n:
        newlen = len(xn) + len(hn) - 1
        xapp = newlen - len(xn)
        happ = newlen - len(hn)
    else:
        if n<len(xn) or n<len(hn):
            raise
        else:
            newlen = n
            xapp = n - len(xn)
            happ = n - len(hn)
    for i in range(xapp):
        xn.append(0)
    for i in range(happ):
        hn.append(0)
    rxn = []
    mxn = []
    for i in range(newlen):
        rxn.append(xn[-i])
    for i in range(newlen):
        rxn.insert(0,rxn.pop())
        x = rxn.copy()
        mxn.append(x)
    mxn.insert(0,mxn.pop())
    xn = np.mat(mxn)
    hn = np.mat(hn)
    return xn*hn.T


if __name__ == '__main__':
    print(pcon(xn,hn))


