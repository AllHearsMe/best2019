import numpy as np

componentList = ['consonant', 'vowel', 'tone']
componentIdx = {s: i for i, s in enumerate(componentList)}
componentCount = len(componentList)

blankChar = '-'

charList = [None]*componentCount
charList[componentIdx['consonant']] = ' 0123456789กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฯๆะาำเแโใไฤฦๅ'
charList[componentIdx['vowel']] = 'ัิีึืุูํ็'
charList[componentIdx['tone']] = '่้๊๋์'
charList = [blankChar + l for l in charList]

charIdx = [{ch: i for i, ch in enumerate(charList[k])} for k in range(componentCount)]
charSet = [set(charList[k]) for k in range(componentCount)]
charCount = [len(charList[k]) for k in range(componentCount)]
toOnehot = [np.eye(charCount[k]) for k in range(componentCount)]

charType = {ch: k for k in range(componentCount) for ch in charList[k]}
charType[blankChar] = -1

onehotIdx = list(np.cumsum(charCount))
onehotOffset = [0]+onehotIdx[:-1]
onehotLen = onehotIdx[-1]
onehotSlices = [slice(s, e) for (s, e) in zip(onehotOffset, onehotIdx)]

def str2idx(s, length=0):
    idx = []
    for ch in s:
        cmp = charType[ch]
        if cmp < 0:
            continue
        elif not idx or cmp == 0 or idx[-1][cmp] > 0:
            idx.append([0]*3)
        idx[-1][cmp] = charIdx[cmp][ch]
    return np.pad(idx, ((0, max(0, length - len(idx))), (0, 0)), 'constant')

def idx2str(idx):
    n = idx.shape[0]
    s = [charList[j][idx[i][j]] for i in range(n) for j in range(componentCount) if idx[i][j] > 0]
    return ''.join(s)

def idx2onehot(idx):
    l = [toOnehot[i][idx[:, i]] for i in range(componentCount)]
    onehot = np.concatenate(l, axis=-1)
    return onehot

def onehot2idx(onehot):
    l = [np.argmax(onehot[..., sl], axis=-1)[..., None] for sl in onehotSlices]
    return np.concatenate(l, axis=-1)

def str2onehot(str, length=0):
    onehot = idx2onehot(str2idx(str))
    return np.pad(onehot, ((0, max(0, length - onehot.shape[0])), (0, 0)), 'constant')

def onehot2str(onehot):
    return idx2str(onehot2idx(onehot))

if __name__ == '__main__':
#     for k in charIdx:
#         print(charIdx[k])
#     idx = str2idx('เป็นมนุษย์สุดประเสริฐเลิศคุณค่า')
#     print(idx)
#     s = idx2str(idx)
#     print(s)
#     onehot = idx2onehot(idx)
#     print(onehot.shape, onehot.sum(axis=1))
#     print(onehot.sum(axis=0))
#     idx2 = onehot2idx(onehot)
#     print(idx2)
    onehot = str2onehot('เป็นมนุษย์สุดประเสริฐเลิศคุณค่า')
    print(onehot.shape)
    s = onehot2str(onehot)
    print(s)