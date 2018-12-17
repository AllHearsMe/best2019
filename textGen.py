import freetype
import numpy as np
import charUtils
import cv2

# TODO: remove safePad if possible, separate position calculation and actual generation
def gen_text_image(s, face, height, width=None, yScatter=0, xScatter=0, maxSlant=0, inclSlantUp=False, safePad=20):
    slotLen = len(charUtils.str2idx(s))
    yPos = np.empty((len(s), 2), dtype=int)
    xPos = np.empty((len(s), 2), dtype=int)
    size = int(height*face.units_per_EM/(face.bbox.yMax-face.bbox.yMin))
    face.set_char_size(size*64)
    yMax, yMin, xMax, xMin = np.ceil(np.array([face.bbox.yMax, face.bbox.yMin, face.bbox.xMax, face.bbox.xMin])/face.units_per_EM*size).astype(int)
    slant = np.random.randint(maxSlant+1)
    slantDir = np.random.choice((-1, 1)) if inclSlantUp else 1
    slantOffset = slant*(slotLen)
    canvas = np.zeros((height+2*yScatter+slantOffset+2*safePad, (xMax-xMin)*(slotLen+1)+2*safePad))
    origin = yMax+yScatter+slantOffset*(slantDir < 0)+safePad
    face.load_char(s[0])
    advance = -face.glyph.bitmap_left-xMin+safePad
    for i, ch in enumerate(s):
        face.load_char(ch)
        glyph = face.glyph
        bitmap = glyph.bitmap
        metrics = glyph.metrics
        pixels = np.array(bitmap.buffer).reshape((bitmap.rows, bitmap.width))
        yOffset = np.random.randint(-yScatter, yScatter+1)
        xOffset = np.random.randint(-xScatter, xScatter+1)
        advance -= xOffset
        yPos[i][0] = origin-glyph.bitmap_top+yOffset
        xPos[i][0] = advance+glyph.bitmap_left+xOffset
        yPos[i][1] = yPos[i][0]+bitmap.rows
        xPos[i][1] = xPos[i][0]+bitmap.width
        box = canvas[yPos[i][0]:yPos[i][1], xPos[i][0]:xPos[i][1]]
        box[:] = np.maximum(pixels, box)
        advance += metrics.horiAdvance // 64
        origin += slant * slantDir * (glyph.bitmap_left >= 0)
    canvas = canvas[safePad:height+slantOffset+safePad, safePad:advance+xMax]
    scale = height / canvas.shape[0]
    canvas = cv2.resize(canvas, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if width:
        if width > canvas.shape[1]:
            canvas = np.pad(canvas, [(0, 0), (0, width - canvas.shape[1])], 'edge')
        else:
            canvas = canvas[:, :width]
    canvas = canvas.astype(int)
    
    xPos = ((xPos - safePad)*scale).astype(int)
    yPos = ((yPos - safePad)*scale).astype(int)
    return canvas, xPos, yPos

def make_labels(s, xPos, length, windowSizes=[20, 30, 40]):
    windowSizes = np.array(windowSizes)
    windowSizes.sort()
    
    types = np.array([charUtils.charType[ch] for ch in s])
    _s = np.array(list(s))
    sByType = np.array([_s[types == i] for i in range(charUtils.componentCount)])
    xLeft = [xPos[types == i, 0] for i in range(charUtils.componentCount)]
    xRight = [xPos[types == i, 1] for i in range(charUtils.componentCount)]
    xPrev = [np.concatenate([[max(0, xr[0]-windowSizes[-1])], xr[:-1]]) for xr in xRight]
    xDiff = [(xr - xp) for xr, xp in zip(xRight, xPrev)]
    xDiff = [np.minimum(xd, windowSizes[-1]) for xd in xDiff]
    
    xBin = [np.searchsorted(windowSizes, xd) for xd in xDiff]
    xBin = [np.minimum(xb, len(windowSizes)-1) for xb in xBin]
#     starts = [np.maximum(np.minimum(xr - windowSizes[xb] + 1, xl), 0) for xb, xl, xr in zip(xBin, xLeft, xRight)]
    starts = [np.clip(xr - windowSizes[xb] + 1, 0, xl) for xb, xl, xr in zip(xBin, xLeft, xRight)]
    
    labels = np.zeros((charUtils.onehotLen, length))
    for t in range(charUtils.componentCount):
        for ch, st, lf in zip(sByType[t], starts[t], xLeft[t]):
            idx = charUtils.onehotIdx[ch]
            labels[idx, st:lf+1] = 1
    
    leftmost = min([xl.min() for xl in xLeft])
    rightmost = max([xr.max() for xr in xRight])
    labels[charUtils.onehotIdx[' '], :np.clip(leftmost-windowSizes[0]+1, 0, starts[0][0])] = 1
    labels[charUtils.onehotIdx[' '], min(rightmost+windowSizes[0], length):] = 1
    for st, ed in zip(xRight[0][:-1], xLeft[0][1:]):
#         In case st > the latter, this line inserts nothing
        labels[charUtils.onehotIdx[' '], st:ed-windowSizes[0]+1] = 1
    
    for sl in charUtils.onehotSlices:
        roi = labels[sl]
        roi[0, roi.max(axis=0) == 0] = 1
    
    return labels