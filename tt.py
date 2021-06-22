import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort
image = cv2.imread('img_9.jpg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

preprocessed_digits = []

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    # print(x)
    # print(y)
    # print(w)
    # print(h)
    # print(" ")
    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    
    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y+h, x:x+w]
    
    # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(digit, (18,18))
    
    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5,4),(5,5)), "constant", constant_values=0)
    if(w>10 and h>10):
        print(w)
        print(h)
        print(" ")
        plt.imshow(padded_digit,cmap="gray")
        plt.show()
        preprocessed_digits.append((padded_digit,x))

def key_sort(val):
    return val[1]

preprocessed_digits.sort(key=key_sort)

# for x in preprocessed_digits:
#     plt.imshow(x[0])
#     plt.show()
# print("\n\n\n----------------Contoured Image--------------------")
plt.imshow(image, cmap="gray")
plt.show()
# inp = np.array(preprocessed_digits)