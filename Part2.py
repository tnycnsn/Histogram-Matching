import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from moviepy.editor import *


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
        hist[j] = np.count_nonzero(patch[:, :, dim] == j)
        #count each value and assign the total # of that value to the (value)th element of hist
    return hist


def hist_match(image, target):
    for i in range(3):
        hist_image = get_hist(image, i)
        #calculate histogram of the ith channel for the image
        cdf_image = hist_image.cumsum() / hist_image.sum()
        #calculate cdf. Note: by dividing to the sum of the histogram, i made histogram to a PDF

        hist_target = get_hist(target, i)
        #calculate histogram of the ith channel for the target image
        cdf_target = hist_target.cumsum() / hist_target.sum()
        #calculate cdf. Note: by dividing to the sum of the histogram, i made histogram to a PDF

        LUT_i = get_LUT(cdf_image, cdf_target)
        #create a Look-Up Table in order to rearrange each pixel's intensity
        image[:,:,i] = np.uint8(LUT_i[image[:,:,i]])
        #finally, map each point according to the Look-Up Table

    return image


main_img_dir = "./Part2/boat/"
all_images = os.listdir(main_img_dir)

target_dir = "./target/Sarap.png"
target_img = cv2.imread(target_dir)

frame_list = []

for i in range(len(all_images)):

    image = cv2.imread(main_img_dir + all_images[i])
    #image is a numpy array with shape (800, 1920, 3)

    image = hist_match(image, target_img)
    #Map via hist_match function
    
    frame_list.append(image[:, :, ::-1])
    #reverse BGR to the RGB
    print("image proceed:", i)


clip = ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('part2_video_boat.mp4', codec='mpeg4')
#write the result image sequence as a video
