import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class CoreFrame:
    def __init__(self, core_width=3, core_height=3):
        self.width = core_width
        self.height = core_height

    def random_mat(self):
        return np.random.random((self.height, self.width))

    def dferf(self):
        return np.array([[-1 / 8, -1 / 8, -1 / 8],                  # 3x3 differential core, not related to frame size yet
                         [-1 / 8,    2,   - 1 / 8],                 # Mathematics work needed
                         [-1 / 8, -1 / 8, -1 / 8]])

    def edge_detect(self):
        return np.array([[-1,   -1,   -1],                          # 3x3 edge_detect core,not related to frame size yet
                         [-1,   8,    -1],                          # Mathematics work needed
                         [-1,   -1,   -1]])

    def itgrf(self):
        return np.array([[1 / 8,    1 / 8,  1 / 8],                  # 3x3 integral core,not related to frame size yet
                         [1 / 8,        0,  1 / 8],                  # Mathematics work needed
                         [1 / 8,    1 / 8,  1 / 8]])


def convolution(img, core, frame):
    height, width, channel = img.shape
    newimg = np.zeros((height + frame.height - 1, width + frame.width - 1, channel))
    newimg[:, :, 0] = signal.convolve(img[:, :, 0], core)                  # concatenate the RGB channels with three 2D arrays
    newimg[:, :, 1] = signal.convolve(img[:, :, 1], core)
    newimg[:, :, 2] = signal.convolve(img[:, :, 2], core)                  # OR signal.fftconvolve(img, core)
    return newimg


def main(name, save=False):
    file_name = './resource/' + str(name) + '.png'
    img = mpimg.imread(file_name)
    frame = CoreFrame()
    core = frame.itgrf()
    print('Convolution Core:', core)
    newimg = convolution(img, core, frame)
    print(newimg.shape)

    plt.figure(figsize=(16, 9), dpi=120)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.imshow(newimg)
    plt.axis('off')
    if save:
        plt.savefig('./output/' + str(name) + '_processed' + '.png', dpi=120, transparent=True)
    plt.show()


if __name__ == '__main__':
    main(4, save=True)
