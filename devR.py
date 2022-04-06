
## [imports]
import cv2 #as cv
import sys

import pyrealsense2 as rs
import numpy as np
#import cv2

import keyboard

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
    
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            (B, G, R) = cv2.split(depth_colormap)
            GRAY = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            #B = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
            #G = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)

            #invert image
            #R = cv2.bitwise_not(R)

            #convert to BGR for display
            R = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
            GRAY = cv2.cvtColor(GRAY, cv2.COLOR_GRAY2BGR)

            
            #images1 = np.hstack((color_image, B, G))
            images2 = np.hstack((color_image, R, GRAY))
            #images = np.hstack((images1, images2))
            

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images2)
        cv2.waitKey(1)
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images2)
        #cv2.waitKey(1)


        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('You Pressed A Key!')
                break  # finishing the loop
        except:
            break  # if user pressed a key other than the given key the loop will break

finally:

    # Stop streaming
    pipeline.stop()

################################################################################

gerandcanny1 = 0
gerandcanny2 = 0
gerandblur = 1

#img1=img2=GRAY
img1=img2=R
#img2=depth_colormap

## [imread]
## [empty]
if img1 is None or img2 is None:
    sys.exit('Could not read the image.')
## [empty]

#get single channel img / or greyscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#save unblurred img
cv2.imwrite('output0.png',img1)

#blur img
img1=cv2.blur(img1,(gerandblur,gerandblur))


#cv2.imwrite('output0.png',img1)
#sys.exit('Blurred image generated!')


#get contours
threshold1 = gerandcanny1
threshold2 = gerandcanny2
# Detect edges using Canny
canny_output = cv2.Canny(img1, threshold1, threshold2 )

#get img1 to BGR
img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

# Find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:10]
cv2.drawContours(img1, contours, -1, (0,0,255), 1)


# loop over our contours
box = []
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    if len(approx) == 4 :
        cv2.drawContours(img1, [approx], -1, (0,255,0), 3)

cv2.imwrite('output.png',img1)
## [imsave]
