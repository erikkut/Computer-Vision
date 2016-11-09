import numpy

def changeContrast(im, level):
    if(level <= -256 & level >= 256):
        print('CONTRAST CAN ONLY BE BETWEEN -255 AND 255, RETURNING ONLY ORIGINAL IMAGE')
        return im

    factor = (259 * (level + 255)) / (255 * (259 - level))

    filler = numpy.zeros(im.shape)
    filler[:,:,:] = 128
    im = im - filler
    im = im * factor
    im = im + filler

    idxLarge = im[:,:,:] > 255
    im[idxLarge] = 255
    idxSmall = im[:,:,:] < 0
    im[idxSmall] = 0

    im = numpy.array(im, 'uint8')
    return im


