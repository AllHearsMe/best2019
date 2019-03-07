import numpy as np


def perlin(x, y):
    """Generate Perlin noise."""
    # permutation table
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi], xf, yf)
    n01 = gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = gradient(p[p[xi+1]+yi], xf-1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b-a)


def fade(t):
    # "6t^5 - 15t^4 + 10t^3"
    # return 6 * t**5 - 15 * t**4 + 10 * t**3
    "3t^2 - 2t^3"
    return t * t * (3. - 2. * t)


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def make_grid(imshape, grid_size):
    """Create gridded linear space used for Perlin noise generation.

    Parameters
    ----------
    imshape : tuple
        Dimensions of the space.
    grid_size : tuple
        Numbers of grid points -- the higher, the finer blobs.
    """
    liny = np.linspace(0, grid_size[0], imshape[0], endpoint=False)
    linx = np.linspace(0, grid_size[1], imshape[1], endpoint=False)
    return np.meshgrid(linx, liny)


def make_noise(grid=None, imshape=(225, 2200, 1), grid_size=(1, 2), random_offset=True, color_range=255):
    """Create Perlin noise, normalized to [0, 255].

    Parameters
    ----------
    grid : tuple of arrays or None (default None)
        Linear space as obtained from calling make_grid. If None, imshape and grid_size will be used to create the grid.
    imshape : tuple (default (225, 2200, 1))
        Dimensions of the space. Only used in case grid is None.
    grid_size : tuple (default (1, 2))
        Numbers of grid points -- the higher, the finer blobs. Only used in case grid is None.
    random_offset : boolean (default True)
        Whether the grid will be randomly shifted along either axis, placing the nodes/blobs in a different location.
    color_range : int (default 255)
        Range of values in target color space.
    """
    x, y = grid if grid else make_grid(imshape, grid_size)
    if random_offset:
        x += np.random.random()
        y += np.random.random()
    p = perlin(x, y)
    p = (p / np.abs(p).max() / 2 + 0.5) * color_range
    return p


# TODO separate the randomization
def add_noise(canvas, noise, noise_weight=0.5):
    """Weighted sum between canvas and noise."""
    return lerp(canvas, noise, noise_weight)


def adjust_lighting(canvas, gain=1, bias=0, color_range=255):
    """Adjust gain and bias, centered on (color_range / 2)."""
    ret = lerp(color_range/2, canvas, gain) + bias
    return np.clip(ret, 0, color_range)


# TODO consider changing to class to save grid_sizes
def random_background(canvas, grid_sizes=[(1, 2)], random_offset=True, noise_weight_range=(0.25, 0.75), color_range=255):
    """Add random background noise to canvas.

    Parameters
    ----------
    canvas : numpy.array
        Original text image.
    grid_sizes : list of tuples (default [(1, 2)])
        List of grid sizes to be randomly chosen from.
    random_offset : boolean (default True)
        Whether the grid will be randomly shifted along either axis, placing the nodes/blobs in a different location.
    noise_weight_range : tuple (default (0.25, 0.75))
        Range of opacity weights when adding noise to canvas.
    color_range : int (default 255)
        Range of values in target color space.
    """
    noise = make_noise(None, canvas.shape,
                       grid_sizes[np.random.choice(len(grid_sizes))], color_range)
    noise_min, noise_max = np.clip(noise_weight_range, 0, 1)
    noise_weight = np.random.uniform(noise_min, noise_max)
    return add_noise(canvas, noise, noise_weight)


def random_lighting(canvas, gain_range=None, bias_range=None, gain_values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], bias_values=[-64, -32, -16, -8, 0, 8, 16, 32, 64], color_range=255):
    """Randomly adjust bias (brightness) and gain (contrast) of canvas.

    Parameters
    ----------
    canvas : numpy.array
        Original text image.
    gain_range : tuple or None (default None)
        Range (min, max), inclusive, from which gain is sampled. If None, sample from gain_values instead.
    bias_range : tuple or None (default None)
        Range (min, max), inclusive, from which bias is sampled. If None, sample from bias_values instead.
    gain_values : list (default [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        List of values from which gain is sampled. Only used when gain_range is None.
    bias_values : list (default [-64, -32, -16, -8, 0, 8, 16, 32, 64])
        List of values from which bias is sampled. Only used when bias_range is None.
    color_range : int (default 255)
        Range of values in target color space.
    """
    gain = np.random.choice(
        gain_values) if gain_range is None else np.random.uniform(*gain_range[:2])
    bias = np.random.choice(
        bias_values) if bias_range is None else np.random.uniform(*bias_range[:2])
    return adjust_lighting(canvas, gain, bias, color_range)


# lerp(max(min_weight, 0), min(max_weight, 1), np.random.random())
