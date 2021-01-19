#!/usr/bin/env python
# coding: utf-8

# ### **GRIP-THE SPARKS FOUNDATION**
# ### **Computer Vision and IOT Internship**
# ### **BY:Anwesha Subhadarshini Dash**
# _________________________________________________________________________
# 
# ## **OPTICAL CHARACTER RECOGNITION(OCR)**
# ### **TASK 1: OPTICAL CHARACTER RECOGNITION**
# 
# #### In this recognition task we will create a character detector which extracts printed or handwritten text from an image or video

# # OPTICAL CHARACTER EXTRACTION USING TESSERACT AND RECOGNITION USING OPENCV
# 
# #### Here the text is extracted from the image using tesseract library  and the character is recognized using open cv

# In[72]:


pip install opencv-python


# In[73]:


pip install pytesseract


# In[210]:


pip install imutils


# In[2]:


import cv2 
import pytesseract 
import numpy as np
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression


# In[3]:


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


# In[5]:


#read the image and covert it to grey scale
image = cv2.imread("quote.png") 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
plt.imshow(image)
  


# In[205]:


#extracting the text using (.image_to_string)
text = pytesseract.image_to_string(image)
print(text)


# In[212]:


#read the image through the path
image=r"C:\Users\DELL\Downloads\task\quote.png"
min_confidence=0.3
width=320
height=320
#read the east detector through the path
east=r"C:\Users\DELL\Downloads\task/frozen_east_text_detection.pb"
#don't read the image directly from the first variable otherwise it will throw "'str' object has no attribute 'copy'"
image = cv2.imread(image)


# In[213]:


#read the original image and it's height and widtha and then set new size for the image and find it's ratio
#resize the image with the new dimensions
original = image.copy()
(H, W) = image.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
(new_W, new_H) = (width, height)
r_W = W / float(new_W)
r_H = H / float(new_H)
# resize the image and grab the new image dimensions
image = cv2.resize(image, (new_W, new_H))
(H, W) = image.shape[:2]


# In[214]:


#loading the east detector and there are two output layers so,we will load the model and find out the layers 
#the output layers derives the boundary box dimensions
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
net = cv2.dnn.readNet(east)
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)


# In[215]:


#scores are the coordinates of the boundary box 
(scores, geometry) = net.forward(layerNames)
#set the rows and columns from the scores for the boundary box which is rectangular in shape and confidence scores 
(num_row, num_col) = scores.shape[2:4]
rects = [] #number of rows and columns in the boundary box
confidences = [] #confidence score


# In[216]:


#setting the coordinates for boundary box
for y in range(0, num_row):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
    for x in range(0, num_col):
        if scoresData[x] < min_confidence:
            continue
        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        
        #deriving height and width of the boundary box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
        #ending box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        #starting box
        startX = int(endX - w)
        startY = int(endY - h)
        #appending boundary box coordinates to list and confidence scores
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])


# In[218]:


#draw the boundary box
for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    cv2.rectangle(original, (startX, startY), (endX, endY), (0, 0, 255), 2)


# In[219]:


original=cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
plt.imshow(original)

