import cv2, numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # IOU: 23,54%
        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        return img

    def histogram_equlization_g(self, img):
        # IOU: 25,12%
        img = cv2.equalizeHist(img)
        
        return img
    
    
    def sharpen(self, img):
        # IOU: 18,75%
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

        return image_sharp

    def increase_brightness(self, img, value=30):
        # IOU: 27,63%
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return img