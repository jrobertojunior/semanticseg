import cv2 as cv
import os


def resize_all_images(path, width, height):
    count = 0
    print("Resizing all images in " + path + " to " + str(width) + "x" + str(height))
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img = cv.imread(path + filename)
            # resize image without interpolation
            resized_img = cv.resize(
                img, (width, height), interpolation=cv.INTER_NEAREST
            )
            cv.imwrite(path + filename, resized_img)
            count += 1

        # else if is directory
        elif os.path.isdir(path + filename):
            resize_all_images(path + filename + "/", width, height)

    

    print("Resized " + str(count) + " images")

def show_image_sizes(path):
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img = cv.imread(path + filename)
            # if shape is different from 3, 256, 256
            if img.shape != (256, 256, 3):
                print(filename + ": " + str(img.shape))
        elif os.path.isdir(path + filename):
            show_image_sizes(path + filename + "/")


# resize_all_images("data/", 256, 256)
show_image_sizes("data/")
