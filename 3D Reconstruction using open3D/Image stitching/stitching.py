import numpy as np
import cv2
import glob
import imutils

def stitch_images(image_paths):
    images = []
    for image in image_paths:
        img = cv2.imread(image)
        images.append(img)
    image_stitcher = cv2.Stitcher_create()
    error, stitched_img = image_stitcher.stitch(images)
    return error, stitched_img

def process_stitched_image(stitched_img, output_filename):
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    min_rectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        min_rectangle = cv2.erode(min_rectangle, None)
        sub = cv2.subtract(min_rectangle, thresh_img)

    contours = cv2.findContours(min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(areaOI)
    stitched_img = stitched_img[y:y + h, x:x + w]

    cv2.imwrite(output_filename, stitched_img)

# Paths to RGB and depth images
rgb_image_paths = glob.glob('unstitchedImages/*.jpg')
depth_image_paths = glob.glob('unstitchedDepthImages/*.png')  # Assuming depth images are stored as PNG

# Stitch RGB images
error_rgb, stitched_rgb_img = stitch_images(rgb_image_paths)
if not error_rgb:
    process_stitched_image(stitched_rgb_img, "stitchedOutputRGBProcessed.png")
    cv2.imshow("Stitched RGB Image Processed", stitched_rgb_img)
    cv2.waitKey(0)
else:
    print("RGB images could not be stitched!")
    print("Likely not enough keypoints being detected!")

# Stitch depth images
error_depth, stitched_depth_img = stitch_images(depth_image_paths)
if not error_depth:
    process_stitched_image(stitched_depth_img, "stitchedOutputDepthProcessed.png")
    cv2.imshow("Stitched Depth Image Processed", stitched_depth_img)
    cv2.waitKey(0)
else:
    print("Depth images could not be stitched!")
    print("Likely not enough keypoints being detected!")

cv2.destroyAllWindows()
