## import packages
import cv2
import numpy as np
import sys

"""
- CLAHE = Contrast Limited Adaptive Histogram Equalization
"""
class CLAHEPreprocessor:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def preprocess(self, image):
        img = image.astype(np.uint8)
        # convert to LAB domain
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return bgr


"""
- SimpleWhiteBalance
"""
class SimpleWhiteBalancePreprocessor:
    def __init__(self, p=0.4):
        self.p = p
    
    def preprocess(self, image):
        wb = cv2.xphoto.createSimpleWB()
        wb.setP(self.p)
        img_wb = wb.balanceWhite(image)
        return img_wb

"""
- Normalize pixels to [0, 1]
"""
class SimpleNormalize:
    def __init__(self, value=255.0):
        self.value = value

    def preprocess(self, image):
        return image / self.value


"""
- Random Satuation - more effective on colorful images
"""
class RandomSaturation:
    """Prob=0.5 to Randomly multiply a factor on Saturation channel
    - Args:
        - lower: lower bound of factor
        - upper: upper bound of factor
    - Returns:
        - image
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def preprocess(self, image):
        # convert to HSV space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype("float")
        # prob = 0.5
        if np.random.randint(0, 2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        # convert back to RGB space
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class RandomHue:
    """Random Hue - change color; harmful in this game
    - Args:
        - delta: bound of random facor to be multiplied to image
    - Returns:
        - processed image
    """
    def __init__(self, delta=5.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def preprocess(self, image):
        # convert to HSV space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype("float")
        if np.random.randint(0, 2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        return image


class RandomLightingNoise:
    """RandomLightingNoise, randomly permute 3 channels 
        - only effective for colorful images
    - Args:
        - image
    - Returns:
        - channel shuffled images
    """

    def __init__(self):
        self.perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0)]

    def preprocess(self, image):
        if np.random.randint(2):
            # 随机选取一个通道的交换顺序，交换图像三个通道的值
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    - Args:
        - swaps (int triple): final order of channels, eg: (2, 1, 0)
    - Returns
        - channel shuffled images
    """
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class Sharpen:
    """Sharpen images
    - Args:
        - k_size
    - Returns:
        - blurred image
    """
    def __init__(self):
        #锐化
        self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) 

    def preprocess(self, image):
        image = cv2.filter2D(image, -1, kernel=self.kernel)
        return image


class RandomCutout:
    """RandomCutOut multi patches from an image
    - Args:
        - n_holes: # of patches to cut out from each image
        - length: # of pixels of each square patch
    - Returns:
        - image with random cutout holes
    """
    def __init__(self, n_holes=2, length=100, pixel_level=False):
        self.n_holes = n_holes
        self.length = length
        self.pixel_level = pixel_level

    def preprocess(self, image):
        H, W, _ = image.shape
        mask = np.ones((H, W), np.float32) 

        for i in range(self.n_holes):
            if self.pixel_level == True:
                n = int(np.sqrt(self.n_holes))
                use_replace = False if n < min(W, H) else True
                x = np.random.choice(W, n, replace=use_replace)
                y = np.random.choice(H, n, replace=use_replace)
                # creat meshgrid
                xv, yv = np.meshgrid(x, y, sparse=False, indexing="xy")
                # expand holes from centers
                xv1 = np.clip(xv - self.length // 2, 0, W)
                xv2 = np.clip(xv + self.length // 2, 0, W)
                yv1 = np.clip(yv - self.length // 2, 0, H)
                yv2 = np.clip(yv + self.length // 2, 0, H)

                # loop over
                for i in range(yv1.shape[0]):
                    for j in range(yv2.shape[1]):
                        y1 = yv1[i][j]
                        y2 = yv2[i][j]
                        x1 = xv1[i][j]
                        x2 = xv2[i][j]
                        mask[y1 : y2, x1 : x2] = 0
                    pass 
            else:
                x, y = np.random.randint(W), np.random.randint(H)
                y1 = np.clip(y - self.length // 2, 0, H)
                y2 = np.clip(y + self.length // 2, 0, H)
                x1 = np.clip(x - self.length // 2, 0, W)
                x2 = np.clip(x + self.length // 2, 0, W)
                # get mask
                mask[y1 : y2, x1 : x2] = 0

        # apply to image
        image = image * np.stack([mask, mask, mask], axis=-1)
        return image


class RandomErasing:
    """            
    p : the probability that random erasing is performed
    s_l, s_h : minimum / maximum proportion of erased area against input image
    r_1, r_2 : minimum / maximum aspect ratio of erased area
    v_l, v_h : minimum / maximum value for erased area
    pixel_level : pixel-level randomization for erased area
    """
    def __init__(self, p=0.5, portion_l=0.02, portion_h=0.4, ar_l=0.3,
            ar_h=1/0.3, val_l=0, val_h=255, pixel_level=False):
        self.p = p
        self.portion_l = portion_l
        self.portion_h = portion_h
        self.ar_l = ar_l
        self.ar_h = ar_h
        self.val_l = val_l
        self.val_h = val_h
        self.pixel_level = pixel_level

    def preprocess(self, image):
        H, W, C = image.shape

        # do not apply random erasing
        p1 = np.random.rand()
        if p1 > self.p:
            return image
        
        while True:
            portion = np.random.uniform(self.portion_l, self.portion_h) * H * W
            ar = np.random.uniform(self.ar_l, self.ar_h)
            w = int(np.sqrt(portion / ar))
            h = int(np.sqrt(portion * ar))
            left = np.random.randint(0, W)
            top = np.random.randint(0, H)
            
            # check 
            if left + w <= W and top + h <= H:
                break

        # check whether to apply at pixel level
        if self.pixel_level:
            c = np.random.uniform(self.val_l, self.val_h, (h, w, C))
        else:
            c = np.random.uniform(self.val_l, self.val_h)

        # return
        image[top : top + h, left: left + w, :] = c
        return image




#################### not very useful ########################

class SingleChannelToGray:
    """extract green channel of dark image & convert it to gray
    - Args:
        - chanDim: which channel to be used
    - Returns:
        - gray image
    """
    def __init__(self, chanDim=1):
        self.chanDim = chanDim

    def preprocess(self, image):
        img_channel = image[:, :, self.chanDim]
        gray =  cv2.merge([img_channel, img_channel, img_channel])
        return gray 

class Reverse:
    def __init__(self):
        pass

    def preprocess(self, image):
        return 255 - image  


class LogTransform:
    def __init__(self, c=42):
        self.c = c

    def preprocess(self, image):
        output = self.c * np.log(1.0 + image)
        output = np.uint8(output + 0.5)
        return output


class Gamma:
    def __init__(self, c=5e-2, gamma=0.8):
        self.c = c
        self.gamma = gamma

    def preprocess(self, image):
        print(image)


        lut = np.zeros(256, dtype=np.float32)
        for i in range(256):
            lut[i] = self.c * i ** self.gamma
        #print(lut)
        
        output_img = cv2.LUT(image, lut) #像素灰度值的映射

        print(output_img)

        output_img = (output_img * 255).astype(np.uint8)
        return output_img


class Deblur:
    """Detect if an image is blurred, if blurred, remove blurry vai Wiener
    filter; 最小均方差（维纳）滤波用来去除含有噪声的模糊图像，
    其目标是找到未污染图像的一个估计，使它们之间的均方差最小，
    可以去除噪声，同时清晰化模糊图像
    - Args:
        - thres: threshold value to judge if an image is blurred
    - Returns
        - clear image
    """
    def __init__(self, thres=200):
        self.thres = thres 
        self.blurred = False

    def preprocess(self, image):
        # convert to gray scale at first
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < self.thres:
            self.blurred = True
        # if blurred, remove blurry via Wiener filter
        if self.blurred:
            from scipy.signal import convolve2d
            from skimage import color, data, restoration

            img = color.rgb2gray(image)
            psf = np.ones((5, 5)) / 25
            img = convolve2d(gray, psf, 'same')
            img += 0.1 * img.std() * np.random.standard_normal(img.shape)

            deconvolved_img = restoration.richardson_lucy(img, psf, 5,
                    clip=False)
#            #deconvolved_img = restoration.wiener(img, psf, 50, clip=False)
#            deconvolved_img, _ = restoration.unsupervised_wiener(img, psf,
#                    clip=False)
#
#            print(deconvolved_img)
            return deconvolved_img
        else:
            return image
