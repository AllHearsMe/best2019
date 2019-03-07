import freetype
import numpy as np
import charUtils
import cv2

def gen_text_image(s, face, height, width=None, yScatter=0, xScatter=0, maxSlant=0, inclSlantUp=False, minScaleFactor=None, safePad=20):
    """Create a text image from the specified string.
    
    Parameters
    ----------
    s : string
        String to be made into an image.
    face : freetype.Face
        Font face the text image will be created with.
    height : int
        Desired height of resulting image. Also used to initialize the font face's size; however, if the resulting image would be too tall, it will be scaled to match this as well.
    width : int or None (default None)
        Desired width of resulting image. Unless None, the resulting image will be cropped or padded to match.
    yScatter, xScatter : int (default 0)
        Maximum number of pixels each character is allowed to randomly shift along the Y- and X-axes.
    maxSlant : int (default 0)
        Maximum number of pixels each character may be offset vertically from the previous character (randomly determined once for the entire image). Does not affect characters that are off the baseline.
    inclSlantUp : boolean (default False)
        Whether upward offsets are allowed to happen.
    minScaleFactor : float in interval (0, 1) or None (default None)
        Minimum size decrease compared to the previous character (randomly determined once for the entire image). If None, characters will not be scaled down at all.
    safePad : int (default 20)
        Deprecated. Amount of pixels to pad the image with, ensuring no characters go out of bound, before cropping back to the intended size.
    """
    yPos = np.empty((len(s), 2), dtype=int)
    xPos = np.empty((len(s), 2), dtype=int)
    slant = np.random.randint(maxSlant+1)
    slantDir = np.random.choice((-1, 1)) if inclSlantUp else 1
    scaleFactor = (np.random.random() * (1 - minScaleFactor) + minScaleFactor) if minScaleFactor else 1
    
    size = height*face.units_per_EM/(face.bbox.yMax-face.bbox.yMin)
    face.set_char_size(int(size*64))
    origin = int(face.bbox.yMax / face.units_per_EM * size)
    advance = 0
    bitmaps = []
    
    # calculate positions
    for i, ch in enumerate(s):
        face.set_char_size(int(size*64))
        face.load_char(ch)
        glyph = face.glyph
        bitmap = glyph.bitmap
        metrics = glyph.metrics
        yOffset = np.random.randint(-yScatter, yScatter+1)
        xOffset = np.random.randint(-xScatter, xScatter+1)
        advance -= xOffset
        yPos[i][0] = origin-glyph.bitmap_top+yOffset
        xPos[i][0] = advance+glyph.bitmap_left+xOffset
        yPos[i][1] = yPos[i][0]+bitmap.rows
        xPos[i][1] = xPos[i][0]+bitmap.width
        pixels = np.array(bitmap.buffer).reshape((bitmap.rows, bitmap.width))
        bitmaps.append(pixels)
        advance += metrics.horiAdvance // 64
        origin += slant * slantDir * (glyph.bitmap_left >= 0)
        size *= scaleFactor

    # create canvas
    yMin = yPos.min()
    xMin = xPos.min()
    if yMin < 0:
        yPos += -yMin
    if xMin < 0:
        xPos += -xMin
    canvasHeight = max(height, yPos.max())
    canvasWidth = xPos.max()
    canvas = np.zeros((canvasHeight, canvasWidth), dtype='uint8')
    
    # paint characters
    for bmp, xp, yp in zip(bitmaps, xPos.tolist(), yPos.tolist()):
        box = canvas[slice(*yp), slice(*xp)]
        box[:] = np.maximum(bmp, box)
        
    scale = height / canvas.shape[0]
    canvas = cv2.resize(canvas, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if width:
        if width > canvas.shape[1]:
            canvas = np.pad(canvas, [(0, 0), (0, width - canvas.shape[1])], 'constant', constant_values=0)
        else:
            canvas = canvas[:, :width]
    
    return canvas, xPos, yPos

def make_labels(s, xPos, length, windowSizes=[20, 30, 40]):
    """Create non-CTC labels. Deprecated."""
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

def get_font_faces(fontlist):
    """Create a list of font faces from a list of paths to font files."""
    return [freetype.Face(f) for f in fontlist]

def make_idx(a, repeats):
    """Calculate the Cartesian product of multiple ranges, for indexing all combinations. Used internally for gen().
    
    Parameters
    ----------
    a : array of ints
        Size of each range to be combined.
    repeats : int
        Amount of times all the indices will be repeated.
    """
    idxs = np.stack(np.meshgrid(*[np.arange(x) for x in a], indexing='ij'), axis=-1).reshape((-1, len(a)))
    idxs = np.tile(idxs, (repeats, 1))
    return idxs

def gen(fontfaces, lines, useAllFonts=False, repeats=1, batch_size=16, imshape=(225, 2200, 1), dataLen=200, maxLabelLen=80, height=150, noiseChance=0.5, lightingChance=0.5, shuffle=True, split=True, auxOutput=False):
    """A generator function that creates text images along with labels. Generated text will be randomly offset along both axes if it is smaller than the specified size.
    
    Parameters
    ----------
    fontfaces : list of freetype.Face
        List of font faces to generate text images with.
    lines : list of strings
        List of text to make images out of.
    useAllFonts : boolean (default False)
        Whether all fonts will be used for each line of text in one epoch. If False, randomly sample a font for each line.
    repeats : int (default 1)
        Amount of times each line of text will be repeated in one epoch.
    batch_size : int (default 16)
        Batch size.
    imshape : tuple (default (225, 2200, 1))
        Shape of image to be created (channels last).
    dataLen : int (default 200)
        Amount of timesteps in one image, presumably corresponding to the output of the model. Required for CTC loss calculation.
    maxLabelLen : int (default 80)
        Maximum timesteps in one line of label text. Required only to keep labels the same size in batches; the actual length of text is still used for CTC loss calculation.
    height : int (default 150)
        Desired height of text.
    noiseChance : float (default 0.5)
        Probability that the text image will contain background noise.
    lightingChance : float (default 0.5)
        Probability that the text image will have its lighting condition (gain/bias) altered.
    shuffle : boolean (default True)
        Whether samples will be randomly shuffled for each epoch.
    split : boolean (default True)
        Whether labels will be split by word component.
    auxOutput : boolean (default False)
        Whether labels include summary labels for auxillary loss.
    """
    if useAllFonts:
        idxs = make_idx([len(fontfaces), len(lines)], repeats)
    else:
        idxs = np.stack([np.empty(len(lines)*repeats, dtype=int),
                         np.tile(np.arange(len(lines)), repeats)],
                        axis=-1)
    x = np.zeros((batch_size,) + imshape, dtype=int)
    y = np.zeros((batch_size, maxLabelLen, charUtils.componentCount), dtype=int)
    xLen = np.zeros((batch_size,), dtype=int)
    yLen = np.zeros((batch_size,), dtype=int)
    loss = [np.zeros((batch_size,)) for _ in range(charUtils.componentCount)]
    while True:
        if not useAllFonts:
            idxs[:, 0] = np.random.randint(len(fontfaces), size=idxs.shape[0])
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(idxs.shape[0]):
            f, l = idxs[i]
            face = fontfaces[f]
            s = lines[l]
            canvas = np.zeros(imshape, dtype='uint8')
            tempCanvas, _, _ = gen_text_image(s, face, height, None, 7, 5, 3, safePad=150)
            tempCanvas = tempCanvas[..., None]
            if tempCanvas.shape[1] > imshape[1]:
                scale = imshape[1] / tempCanvas.shape[1]
                tempCanvas = cv2.resize(tempCanvas, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)[..., None]
            yOffset = np.random.randint(imshape[0]-tempCanvas.shape[0]+1)
            xOffset = np.random.randint(imshape[1]-tempCanvas.shape[1]+1)
            canvas[yOffset:yOffset+tempCanvas.shape[0],
                   xOffset:xOffset+tempCanvas.shape[1]] = tempCanvas
            x[i%batch_size] = (255-canvas)
            if np.random.random() < noiseChance:
                x[i%batch_size] = noiseUtils.random_noise(x[i%batch_size])
            if np.random.random() < lightingChance:
                x[i%batch_size] = noiseUtils.random_lighting(x[i%batch_size], gain_range=(0.5, 1.0))
            xLen[i%batch_size] = dataLen
            
            encoded_labels = charUtils.str2idx(s, split=split)
            y[i%batch_size] = np.pad(encoded_labels, [(0, max(0, maxLabelLen - encoded_labels.shape[0])), (0, 0)], 'constant')
            yLen[i%batch_size] = encoded_labels.shape[0]
            
            if (i+1)%batch_size == 0:
                inp = [x, y, xLen, yLen]
                outp = (loss + [charUtils.idx2auxout(y)]) if auxOutput else loss
                yield inp, outp
        if idxs.shape[0]%batch_size != 0:
            inp = [x, y, xLen, yLen]
            outp = (loss + [charUtils.idx2auxout(y)]) if auxOutput else loss
            yield [z[:idxs.shape[0]%batch_size] for z in inp], [z[:idxs.shape[0]%batch_size] for z in outp]

def combine_generators(gens, steps, shuffle=True):
    """Combines the outputs of multiple data generators."""
    assert len(gens) == len(steps)
    idxs = np.repeat(np.arange(len(gens)), steps)
    while True:
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(idxs.shape[0]):
            yield next(gens[idxs[i]])