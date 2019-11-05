import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from moviepy.editor import *


main_img_dir = "./Part1/breakdance/"
main_seg_dir = "./Part1/breakdance_seg/"
all_images = os.listdir(main_img_dir)

frame_list = []

for i in range(len(all_images)):
    image = cv2.imread(main_img_dir + all_images[i])
    #image is a numpy array with shape (800, 1920, 3)
    seg = cv2.imread(main_seg_dir + all_images[i].split('.')[0] + '.png', cv2.IMREAD_GRAYSCALE)
    #Seg is the segmentation map of the image with shape (800, 1920)

    seg = (seg == 38)
    #create a mask where seg equals to 38


    #image[seg, 1:] = image[seg, 1:] / 4
    image[seg, 1:3] = 0
    #canceled out the red and green channels where seg map eq. to the 38

    frame_list.append(image[:, :, ::-1])
    #reverse BGR used by OpenCV to RGB

    #cv2.imshow("image", image)
    #cv2.waitKey(40)
    #in order to check the results i used imshow

clip = ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('part1_video_breakdance.mp4', codec='mpeg4')
#write the result image sequence as a video
