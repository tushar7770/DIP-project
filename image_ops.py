import cv2
import numpy as np
import math
from scipy import ndimage

# resize image pass image and a scaling factor like 0.5 , 2 ,3
def resize_img(img,scaling_factor):

    def bilinear_interpolation(img, scaling_factor):
        original_h, original_w, _ = img.shape
        h, w = int(scaling_factor * original_h), int(scaling_factor * original_w)
        new_img = np.zeros([h, w, 3], dtype=np.uint8)

        x_ratio = float(original_w - 1) / (w - 1)
        y_ratio = float(original_h - 1) / (h - 1)

        for i in np.arange(h):
            for j in np.arange(w):
                x = float(j * x_ratio)
                y = float(i * y_ratio)

                xl, yl = int(math.floor(x)), int(math.floor(y))
                xh, yh = int(math.ceil(x)), int(math.ceil(y))

                a = x - xl
                b = y - yl

                p1 = img[yl, xl, :]
                p2 = img[yl, xh, :]
                p3 = img[yh, xl, :]
                p4 = img[yh, xh, :]

                a1 = p1 * (1 - a) + p2 * a
                b1 = p3 * (1 - a) + p4 * a
                pixel_val = a1 * (1 - b) + b1 * b
                new_img[i, j, :] = pixel_val

        return new_img
    
    return bilinear_interpolation(img,scaling_factor)

# adding salt and pepper noise pass image mean and variance
def salt_and_pepper_noise(img,mean,var):
    std=math.sqrt(var)
    gaussian_noise=np.random.normal(mean,std,img.shape).astype(np.uint8)
    noisy_image=cv2.add(img,gaussian_noise)
    return noisy_image

# to change brightness or basically the bits used to represent the image
def brightness(img,upper):

    def intensityRange(img, upper):
        arr = np.array(img)
        max_val = np.max(arr, axis=(0, 1)) 
        min_val = np.min(arr, axis=(0, 1))  
        arr = (arr - min_val) / (max_val - min_val)
        arr = arr * upper
        arr = np.clip(arr, 0, upper)  
        return arr.astype(np.uint8)

    return intensityRange(img,upper)

# rotate image pass image and angle in anticlockwise direction
def rotate(img,angle):

    def shear(theta,x,y):
        '''
        [x_new]  == |1  -tan(𝜃/2) |  |1        0|  |1  -tan(𝜃/2) | [x] == | cos𝜃   sin𝜃 | [x]
        [y_new]     |0      1     |  |sin(𝜃)   1|  |0      1     | [y]    | -sin𝜃  cos𝜃 | [y]
        '''
        tangent=math.tan(theta/2)
        new_x=round(x-y*tangent)
        new_y=y
        
        new_y=round(new_x*math.sin(theta)+new_y)     

        new_x=round(new_x-new_y*tangent)             
        
        return new_x,new_y

    def rotation(rotation_angle_clockwise,img):
        rotation_angle_clockwise=math.radians(rotation_angle_clockwise)
        cosTheta=math.cos(rotation_angle_clockwise)
        sinTheta=math.sin(rotation_angle_clockwise)
        height,width,d=img.shape


        new_height=round(abs(height*cosTheta)+abs(width*sinTheta))+1
        new_width=round(abs(width*cosTheta)+abs(height*sinTheta))+1

        output=np.zeros((new_height,new_width,d))

        #centre of original image
        original_centre_height=round(((height+1)/2)-1)
        original_centre_width=round(((width+1)/2)-1)

        #centre of rotated image
        new_centre_height=round(((new_height+1)/2)-1)
        new_centre_width=round(((new_width+1)/2)-1)

        for i in range(height):
            for j in range(width):

                y=height-1-i-original_centre_height
                x=width-1-j-original_centre_width

                new_x,new_y=shear(rotation_angle_clockwise,x,y)

                new_x=new_centre_width-new_x
                new_y=new_centre_height-new_y

                if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                    output[new_y,new_x,:]=img[i,j,:]

        return output.astype(np.uint8)
    
    return rotation(-1*angle,img)

# contrast enhacement using histogram equalization
def contrast_enhancement(img):

    def histogramEqualisation(img):
        b, g, r = cv2.split(img)
        h, w, _ = img.shape
        
        for channel in [b, g, r]:
            count = [0] * 256
            for i in range(h):
                for j in range(w):
                    count[channel[i, j]] += 1

            p = np.array(count) / (h * w)
            cf = np.cumsum(p)
            map_img = np.uint8(255 * cf)

            for i in range(h):
                for j in range(w):
                    channel[i, j] = map_img[channel[i, j]]

        return cv2.merge((b, g, r))
    
    return histogramEqualisation(img)

# smooth image using gaussian filter pass image and  filter size
def smooth_image(img,size):
    return  cv2.GaussianBlur(img, (size, size), 0)

# remove s and p noise using the median filter pass img and filter size
def remove_salt_and_pepper_noise(img,size):
    return ndimage.median_filter(img, size=size)

# edge detection using canny
def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = smooth_image(gray,5)
    edges = cv2.Canny(gray_blur, 50, 150)
    return edges

#morphological operationn for enhancing text pass image and size of structuring element , iteraton for opening operation
def text_enhancement(img,size,iteration):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to convert to binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Define the structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iteration)

    return opened.astype(np.uint8)

# face detection using haar cascade
def face_detection(img):

    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    print(faces)
    # Draw a square around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (47, 78, 31), 3)
    
    return img

# img=cv2.imread("./abn.png")

# eql=face_detection(img)
# cv2.imshow('testing',eql)