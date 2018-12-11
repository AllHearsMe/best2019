import freetype
import numpy as np
import charUtils
import cv2

def gen_text_image(s, face, height, width=None, yScatter=0, xScatter=0, maxSlant=0, inclSlantUp=False):
    slotLen = len(charUtils.str2idx(s))
    yPos = np.empty((len(s), 2), dtype=int)
    xPos = np.empty((len(s), 2), dtype=int)
    size = int(height*face.units_per_EM/(face.bbox.yMax-face.bbox.yMin))
    face.set_char_size(size*64)
    yMax, yMin, xMax, xMin = np.ceil(np.array([face.bbox.yMax, face.bbox.yMin, face.bbox.xMax, face.bbox.xMin])/face.units_per_EM*size).astype(int)
    slant = np.random.randint(maxSlant+1)
    slantDir = np.random.choice((-1, 1)) if inclSlantUp else 1
    slantOffset = slant*(slotLen)
    canvas = np.zeros((height+2*yScatter+slantOffset, (xMax-xMin)*(slotLen+1)))
    origin = yMax+yScatter+slantOffset*(slantDir < 0)
    face.load_char(s[0])
    advance = -face.glyph.bitmap_left-xMin
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
    canvas = canvas[:height+slantOffset, :advance+xMax]
    scale = height / canvas.shape[0]
    canvas = cv2.resize(canvas, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if width:
        if width > canvas.shape[1]:
            canvas = np.pad(canvas, [(0, 0), (0, width - canvas.shape[1])], 'edge')
        else:
            canvas = canvas[:, :width]
    canvas = canvas.astype(int)
    
    xPos = (xPos*scale).astype(int)
    yPos = (yPos*scale).astype(int)
    return canvas, xPos, yPos

def make_labels(s, xPos, length, windowSizes=[20, 30, 40]):
    windowSizes = np.array(windowSizes)
    xDiff = np.diff(xPos, axis=1).flatten()
    xBin = np.searchsorted(windowSizes, xDiff)
    xBin = np.minimum(xBin, len(windowSizes)-1)
    widths = np.maximum(windowSizes[xBin] - xDiff, 0) + 1
    
    labels = np.zeros((charUtils.onehotLen, length))
    for ch, x, w in zip(s, xPos[:, 0], widths):
        t = charUtils.charType[ch]
        idx = charUtils.onehotIdx[ch]
        labels[idx, x:x+w] = 1
    
    for sl in charUtils.onehotSlices:
        roi = labels[sl]
        roi[0, roi.max(axis=0) == 0] = 1
    
    return labels