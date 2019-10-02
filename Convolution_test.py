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


def convolution(img, core, frame, discrete):
    height, width, channel = img.shape
    newimg = np.zeros((height + frame.height - 1, width + frame.width - 1, channel))
    new_R = signal.convolve(img[:, :, 0], core)       # OR signal.fftconvolve(img, core)
    new_G = signal.convolve(img[:, :, 1], core)
    new_B = signal.convolve(img[:, :, 2], core)
    empchan = np.zeros((height + frame.height - 1, width + frame.width - 1))
    if discrete:
        imgB, imgG, imgR = newimg.copy(), newimg.copy(), newimg.copy()
        imgR[:, :, 0], imgG[:, :, 0], imgB[:, :, 0] = new_R, empchan, empchan
        imgR[:, :, 1], imgG[:, :, 1], imgB[:, :, 1] = empchan, new_G, empchan
        imgR[:, :, 2], imgG[:, :, 2], imgB[:, :, 2] = empchan, empchan, new_B
        newimg = np.concatenate((imgR,imgG,imgB), axis=1)
    else:
        newimg[:, :, 0] = new_R
        newimg[:, :, 1] = new_G
        newimg[:, :, 2] = new_B

    return newimg


def main(name, save=False, discrete=True):
    file_name = './resource/' + str(name) + '.png'
    img = mpimg.imread(file_name)
    frame = CoreFrame()
    core = frame.edge_detect()
    print('Convolution Core:', core)
    newimg = convolution(img, core, frame,discrete)
    print(newimg.shape)
    H, W, C =newimg.shape
    H, W = H/120, W/120
    plt.figure(figsize=(W, H), dpi=120)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.imshow(newimg)
    plt.axis('off')
    if save:
        plt.savefig('./output/' + str(name) + '_processed' + '.png',  transparent=True)
        print("Picture saved as :"+'./output/' + str(name) + '_processed' + '.png')
    plt.show()


if __name__ == '__main__':
    main(5, save=True)
