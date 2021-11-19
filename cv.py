import cv2 as cv
import os

input = cv.imread(
    "data/headsegmentation_dataset_ccncsa/labels/female03/headrende0002.png"
)

boxes = [
    (0, 65, 253, 214),
    (102, 122, 34, 29),
    (98, 157, 41, 19),
    (89, 116, 17, 5),
    (130, 117, 15, 5),
    (134, 98, 18, 10),
    (82, 99, 23, 8),
    (69, 114, 5, 28),
    (41, 43, 146, 169),
]

cv.imshow("input", input)
i = 0
for box in boxes:
    left, top, width, height = box
    # crop input image
    cropped = input[top : top + height, left : left + width]


    cv.imshow("cropped " + str(i), cropped)
    i += 1
    cv.waitKey(0)


def resize_image(image, width, height):
    return cv.resize(image, (width, height))

# recursevly resize all images from folder
# for filename in os.listdir(path):
#     if filename.endswith(".png"):
#         image = cv.imread(path + filename)
#         resized = resize_image(image, width, height)
#         cv.imwrite(path + filename, resized)


def resize_images_on_dir(path):
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            image = cv.imread(path + filename)
            resized = resize_image(image, width, height)
            cv.imwrite(path + filename, resized)