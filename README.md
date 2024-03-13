# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import numpy module as np and pandas as pd.
### Step 2: 
Assign the values to variables in the program.
### Step 3: 
Get the values from the user appropriately.
### Step 4: 
Continue the program by implementing the codes of required topics.
### Step 5: 
Thus the program is executed in google colab.

## Program:

#### Developed By : Priyanka A
#### Register Number : 212222230113

### Installing OpenCV , importing necessary libraries and displaying images  

```py
# Install OpenCV library
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images 
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

```

#### (i) Image Translation
```py
# Load an image from URL or file path
image_url = '1.jpg'  
image = cv2.imread(image_url)

# Define translation matrix
tx = 50  # Translation along x-axis
ty = 30  # Translation along y-axis
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])  # Create translation matrix

# Apply translation to the image
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# Display original and translated images
print("Original Image:")
show_image(image)
print("Translated Image:")
show_image(translated_image)
```

#### (ii) Image Scaling
```py

# Load an image from URL or file path
image_url = '2.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)


# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis


# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)

```




#### (iii) Image shearing
```py
# Load an image from URL or file path
image_url = '3.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)

```



#### (iv) Image Reflection

```py
# Load an image from URL or file path
image_url = '4.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

```

##### (a) → Reflecting Horizontally

```py
# Display original and reflected images

show_image(image)
print("↑ Original Image")
show_image(reflected_image_horizontal)
print("↑ Reflected Horizontally")
```

##### (b) → Reflected Vertically

```py
show_image(image)
print("↑ Original Image")
show_image(reflected_image_vertical)
print("↑ Reflected Vertically")

```

##### (c) → Reflecting Horizontally & Vertically
```py

show_image(image)
print("↑ Original Image")
show_image(reflected_image_both)
print("↑ Reflected Both")

```

### (v) Image Rotation

```py
# Load an image from URL or file path
image_url = '5.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)


# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)

```



### (vi) Image Cropping

```py
# Load an image from URL or file path
image_url = '6.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)

```


## Output:

### (i) Image Translation

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/9403cc4f-659b-4d45-a23b-7eece3d5a2ed) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/ba41546f-e75a-498b-9761-22152f8ca355)

### (ii) Image Scaling

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/a2ce159a-d45b-46ca-88bd-bca1de8bd52d) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/29b73d78-bb84-4e4c-9418-26021f2da065)

### (iii) Image shearing

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/78452bbf-b9a9-4efd-bf7e-4bcc8ba987e3) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/3fda3129-34b2-4a69-9963-802b6b102230)

### (iv) Image Reflection

#### Reflecting Horizontally

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/a5c9b71d-838b-44bc-9e53-30417158e7ac) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/22f9d64d-dfb2-4295-9c10-dd17b166f165)

#### Reflecting Vertically

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/f6a75d44-46f6-47fb-b6e9-c077c463d6cc) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/64f90c2c-e275-451e-9c39-0806381e1542)

#### Reflecting Horizontally & Vertically

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/d906cac6-d59b-434f-870d-646b32afcd8b) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/f27b5c4d-7a9b-48aa-b064-304c20abcefc)

### (v) Image Rotation

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/949c9219-14a4-4b31-8ded-1f360e6afbac) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/ac54593f-6135-4e2f-82c5-bebbfda7da08)

### (vi) Image Cropping

![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/d48c3b16-b51b-4f6c-88d8-85e72aaedd3a) ![image](https://github.com/PriyankaAnnadurai/IMAGE-TRANSFORMATIONS/assets/118351569/197f77c5-f577-4da0-8d69-18d1f75343c6)

## Result: 

### Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
