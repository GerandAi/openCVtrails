## [imports]
import cv2 #as cv
import sys

gerandcanny1 = 55
gerandcanny2 = 255
gerandblur = 155

#from __future__ import print_function
#import cv2 as cv
#import numpy as np
#import random as rng

## [imports]
## [imread] (READING AS A GREYSCALE IMAGE)
img1 = cv2.imread('pi.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('pi.jpg')
## [imread]
## [empty]
if img1 is None or img2 is None:
    sys.exit('Could not read the image.')
## [empty]
## [imshow]
#cv.imshow("Display window", img)
#k = cv.waitKey(0)
## [imshow]
## [imsave]
#if k == ord("s"):
#    cv.imwrite('img0206.png',img)

# get contours
#(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

#blur img
img=cv2.blur(img1,(gerandblur,gerandblur))

#cv2.imwrite('output.png',img)
#sys.exit('Blurred image generated!')


#get contours
threshold1 = gerandcanny1
threshold2 = gerandcanny2
# Detect edges using Canny
canny_output = cv2.Canny(img1, threshold1, threshold2 )


# Find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:10]
cv2.drawContours(img2, contours, -1, (0,0,255), 1)
area1 = cv2.contourArea(contours[1])
area2 = cv2.contourArea(contours[2])
area3 = cv2.contourArea(contours[3])
print(area1)
print(area2)
print(area3)

# loop over our contours
i = 1
box = []
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    if len(approx) == 4 and i <= 3 and area1 - area2 <= 20000 :
        #print(approx)
        box.append(approx)
        cv2.drawContours(img2, [approx], -1, (0,255,0), 2)
        i = i+1

    if len(approx) == 4 and i > 3 :
        #print(approx)
        cv2.drawContours(img2, [approx], -1, (255,0,0), 2)
        i = i+1



##box1 = box[0]
##box2 = box[1]
#print(box1)
#print(box2)


##point11=box1[0][0]
##point12=box1[1][0]
##point13=box1[2][0]
##point14=box1[3][0]
##point21=box2[0][0]
##point22=box2[1][0]
##point23=box2[2][0]
##point24=box2[3][0]

##print(point11)
##print(point12)
##print(point13)
##print(point14)
##print(point21)
##print(point22)
##print(point23)
##print(point24)
        


#xpick = (point11[0]+point12[0]+point13[0]+point14[0]+point21[0]+point22[0]+point23[0]+point24[0]) / 8
#ypick = (point11[1]+point12[1]+point13[1]+point14[1]+point21[1]+point22[1]+point23[1]+point24[1]) / 8
#cv2.circle(img2, (xpick,ypick), radius=10, color=(0, 255, 255), thickness=-1)
#print('pick point is: ')
#print(xpick, ypick)
cv2.imwrite('output1.png',img2)
## [imsave]
