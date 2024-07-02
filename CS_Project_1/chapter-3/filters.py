import cv2
import numpy

import utils


# h - selects histogram equalization
def histeq(image):
    
    # Part 1: get the shape of the greyscale image
    img = image.astype(numpy.float32)
    (nrow,ncol) = img.shape

    # Part 2: Compute the histogram of the image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # find the cumulative distribution function with cumsum() method
    cdf = numpy.cumsum(hist)

    # Part 3: Perform the equalization
    c = nrow*ncol / 255.0 # need to be float here, or integer division
    equal_hist = (1/c) * (cdf / cdf[-1])  # this will give us the equalized histogram

    # Flatten the image to a 1D array
    image_flattened = image.flatten()

    # Use numpy's interpolation function to map the old values to the new ones
    image_equalized_flattened = numpy.interp(image_flattened, numpy.arange(len(equal_hist)), equal_hist)

    # Reshape the equalized image back to the original shape
    image_equalized = image_equalized_flattened.reshape(image.shape)
    
    return image_equalized


# s - selects a 5 × 5 smoothing filter
def smooth(image, intensity):

    # Define a 5x5 smoothing filter
    smoothing_filter = numpy.ones((5, 5), numpy.float32) / (intensity * 1.0)

    # Apply the filter
    smoothed_image = cv2.filter2D(image, -1, smoothing_filter)
    
    return smoothed_image


# u - selects a 5×5 unsharp filter
def unsharp(image):
    # Define a 5x5 unsharp filter
    unsharp_filter = numpy.array([[-1, -1, -1, -1, -1],
                            [-1,  2,  2,  2, -1],
                            [-1,  2,  8,  2, -1],
                            [-1,  2,  2,  2, -1],
                            [-1, -1, -1, -1, -1]])

    # Normalize the filter to make the sum of all the values equal to 1
    unsharp_filter = unsharp_filter / numpy.sum(unsharp_filter)

    # Apply the filter
    unsharp_image = cv2.filter2D(image, -1, unsharp_filter)
    
    return unsharp_image


# e - selects an edge detector filter of your choice
def sobel(image):
    # Apply Sobel filters
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine Sobel filters
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
    return sobel_combined


def recolorRC(src, dst):
    """Simulate conversion from BGR to RC (red, cyan).

    The source and destination images must both be in BGR format.

    Blues and greens are replaced with cyans. The effect is similar
    to Technicolor Process 2 (used in early color movies) and CGA
    Palette 3 (used in early color PCs).

    Pseudocode:
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r

    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)


def recolorRGV(src, dst):
    """Simulate conversion from BGR to RGV (red, green, value).

    The source and destination images must both be in BGR format.

    Blues are desaturated. The effect is similar to Technicolor
    Process 1 (used in early color movies).

    Pseudocode:
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r

    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    """Simulate conversion from BGR to CMV (cyan, magenta, value).

    The source and destination images must both be in BGR format.

    Yellows are desaturated. The effect is similar to CGA Palette 1
    (used in early color PCs).

    Pseudocode:
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r

    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


def blend(foregroundSrc, backgroundSrc, dst, alphaMask):

    # Calculate the normalized alpha mask.
    maxAlpha = numpy.iinfo(alphaMask.dtype).max
    normalizedAlphaMask = (1.0 / maxAlpha) * alphaMask

    # Calculate the normalized inverse alpha mask.
    normalizedInverseAlphaMask = \
        numpy.ones_like(normalizedAlphaMask)
    normalizedInverseAlphaMask[:] = \
        normalizedInverseAlphaMask - normalizedAlphaMask

    # Split the channels from the sources.
    foregroundChannels = cv2.split(foregroundSrc)
    backgroundChannels = cv2.split(backgroundSrc)

    # Blend each channel.
    numChannels = len(foregroundChannels)
    i = 0
    while i < numChannels:
        backgroundChannels[i][:] = \
            normalizedAlphaMask * foregroundChannels[i] + \
            normalizedInverseAlphaMask * backgroundChannels[i]
        i += 1

    # Merge the blended channels into the destination.
    cv2.merge(backgroundChannels, dst)


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)."""

    def __init__(self, vFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        srcFlatView = numpy.ravel(src)
        dstFlatView = numpy.ravel(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView,
                               dstFlatView)

class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""

    def __init__(self, vPoints, dtype=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints),
                             dtype)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""

    def __init__(self, vFunc=None, bFunc=None, gFunc=None,
                 rFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""

    def __init__(self, vPoints=None, bPoints=None,
                 gPoints=None, rPoints=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)

class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0,20),(255,235)],
            gPoints=[(0,0),(56,39),(208,226),(255,255)],
            rPoints=[(0,0),(56,22),(211,255),(255,255)],
            dtype=dtype)

class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0,0),(23,20),(157,173),(255,255)],
            bPoints=[(0,0),(41,46),(231,228),(255,255)],
            gPoints=[(0,0),(52,47),(189,196),(255,255)],
            rPoints=[(0,0),(69,69),(213,218),(255,255)],
            dtype=dtype)

class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0,0),(35,25),(205,227),(255,255)],
            gPoints=[(0,0),(27,21),(196,207),(255,255)],
            rPoints=[(0,0),(59,54),(202,210),(255,255)],
            dtype=dtype)

class BGRVelviaCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0,0),(128,118),(221,215),(255,255)],
            bPoints=[(0,0),(25,21),(122,153),(165,206),(255,255)],
            gPoints=[(0,0),(25,21),(95,102),(181,208),(255,255)],
            rPoints=[(0,0),(41,28),(183,209),(255,255)],
            dtype=dtype)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)

class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)

class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]])
        VConvolutionFilter.__init__(self, kernel)



