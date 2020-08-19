from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import glob


def compare_images(image1, image2, title):
    s = ssim(image1, image2)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("SSIM: %.2f" % (s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image1, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image2, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()
    return s

s1=0
a=glob.glob("images/*")
i1=input("Enter image you want to scan:")#enter with directory
for i2 in a:
    print("For:" + i2)
    img1 = cv2.imread(i1)
    img2 = cv2.imread(i2)
    img1=cv2.resize(img1, (600, 800))
    img2=cv2.resize(img2,(600, 800))
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# initialize the figure
    #fig = plt.figure("Images")
    images = ("Image 1", img1), ("Image 2", img2)

    # compare the images
    s=compare_images(img1, img2, "Image1 vs. Image2")
    if(s>s1):
        s1=s

print("highest match:",s1*100,"%")
