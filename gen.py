import cv2
import numpy as np
import os

path = './images/'
images = np.stack([cv2.imread(path+i) for i in os.listdir(path)])
print(images.shape)
video = cv2.VideoWriter('seg.mp4', 0, 30, (511, 511)) 
for image in images:
	video.write(image) 
cv2.destroyAllWindows() 
video.release() 