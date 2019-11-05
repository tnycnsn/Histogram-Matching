import cv2
from matplotlib import pyplot as plt
import os
from moviepy.editor import *
import numpy as np


def get_LUT(cdf_img, cdf_target):
    m = cdf_img.shape[0]
    #range of LUT: [0, m]
    LUT = np.zeros(m)
    g_t = 0
    #target domain pointer
    for g_i in range(m):
        while cdf_target[g_t] < cdf_img[g_i] and g_t < 255 :
            g_t += 1
        LUT[g_i] = g_t
    return LUT


def get_hist(patch, dim):
    #patch: expected 3 dim. array which is [height, width, channel]
    #dim: indicates channel of image
    hist = np.zeros((256, 1))
    for j in range(256):
        hist[j] = np.count_nonzero(patch[:, dim] == j)
        #since each patch determined with mask each channel became flatten so patches became 2D
        #count each value and assign the total # of that value to the (value)th element of hist
    return hist


def hist_match(image, target):
    for i in range(3):
        hist_image = get_hist(image, i)
        #hist_image = cv2.calcHist([image], [i], None, [256], [0, 256])
        cdf_image = hist_image.cumsum() / hist_image.sum()
        #calculate cdf of each ith channel of image

        hist_target = get_hist(target, i)
        #hist_target = cv2.calcHist([target], [i], None, [256], [0, 256])
        cdf_target = hist_target.cumsum() / hist_target.sum()

        LUT_i = get_LUT(cdf_image, cdf_target)
        image[:, i] = np.uint8(LUT_i[image[:, i]])
        #(we feed image parameter with patches) --> since each patch determined with mask each channel became flatten so patches became 2D
    return image


main_img_dir = "./Part3/walking/"
main_seg_dir = "./Part3/walking_seg/"
all_images = os.listdir(main_img_dir)

target_dir = "./target2/"
all_targets = os.listdir(target_dir)
np.random.shuffle(all_targets)
#shuffle to randomize the selection

seg = cv2.imread(main_seg_dir + all_images[0].split('.')[0] + '.png', cv2.IMREAD_GRAYSCALE)
value_list = np.unique(seg)
#contains values of each different seg

frame_list = []

#print(all_targets[:len(value_list)], "\n", value_list)

for i in range(len(all_images)):

    image = cv2.imread(main_img_dir + all_images[i])
    #image is a numpy array with shape (800, 1920, 3)
    seg = cv2.imread(main_seg_dir + all_images[i].split('.')[0] + '.png', cv2.IMREAD_GRAYSCALE)

    for j, value in enumerate(value_list):
        mask_i = (seg == value)
        target_img = cv2.imread(target_dir + all_targets[j])
        image[mask_i, :] = hist_match(image[mask_i, :], target_img)

    frame_list.append(image[:, :, ::-1])
    #reverse BGR used by OpenCV to RGB
    print("image proceed:", i)


clip = ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('part3_video_walking.mp4', codec='mpeg4')
#write the result image sequence as a video
