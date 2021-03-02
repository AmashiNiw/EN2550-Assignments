# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # EN2550 - Assignment 01 - 180433E
# ## Question 1.
# ### Some Useful Modules and Functions to Display Images and Create Salt and Pepper Noise

# %%
# importing necessary modules

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Display two images side by side

def show_img(img, img2, size = 16):
    
    fig, axes  = plt.subplots(1,2, sharex='all', sharey='all', figsize=(size,size))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Modified')
    axes[1].set_xticks([]), axes[1].set_yticks([])
    plt.show()

# Transform an image to one with salt and pepper noise

def noisy(noise_typ,image):
    if noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.2
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out

# %% [markdown]
# ## Question 1. (a) Histogram Calculation

# %%
# Histogram of the Color Image

img = cv.imread('./a01images/aragon.jpg', cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
color = ('r', 'g', 'b')
for i, c in enumerate(color):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color = c)
    plt.xlim([0,256])
plt.title('Histogram of the Image')
plt.show()

# %% [markdown]
# ## Question 1. (b) Histogram Equilizing

# %%
# Histogram Equilization of a Colour Image

'''Split the 3 Color Channels'''
R, G, B = cv.split(img)

'''Process Them Seperately'''
output1_R = cv.equalizeHist(R)
output1_G = cv.equalizeHist(G)
output1_B = cv.equalizeHist(B)

'''Merge to Get The Equilized Image'''
equ = cv.merge((output1_R, output1_G, output1_B))

plt.hist(output1_R.flatten(),256,[0,256], color = 'r')
plt.hist(output1_G.flatten(),256,[0,256], color = 'g')
plt.hist(output1_B.flatten(),256,[0,256], color = 'b')

plt.xlim([0,256])
plt.title('Histogram of the Equalized Image')
plt.show()

show_img(img,equ)

# %% [markdown]
# ## Question 1. (c) Intensity Tranformations

# %%
# Identity Transform

transform = np.arange(0, 256).astype('uint8')

fig, ax = plt.subplots()
ax.plot(transform)
ax.set_xlabel(r'Input, $f(\mathbf{x})$')
ax.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_aspect('equal')
plt.show()

image_transformed = cv.LUT(img, transform)
show_img(img,image_transformed)


# %%
# Negative Transform

neg_transform = np.arange(255,-1, -1).astype('uint8')
image_neg_transformed = cv.LUT(img, neg_transform)

fig, ax = plt.subplots()
ax.plot(neg_transform)
ax.set_xlabel(r'Input, $f(\mathbf{x})$')
ax.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_aspect('equal')
plt.show()

show_img(img, image_neg_transformed)


# %%
# Intensity Window Transform

c = np.array([(100, 60), (150, 180)])

t1 = np.linspace(0, c[0,1], c[0,0] + 1 - 0).astype('uint8')
t2 = np.linspace(c[0,1] + 1, c[1,1], c[1,0] - c[0,0]).astype('uint8')
t3 = np.linspace(c[1,1] + 1, 255, 255 - c[1,0]).astype('uint8')

window_transform = np.concatenate((t1, t2), axis=0).astype('uint8')
window_transform = np.concatenate((window_transform, t3), axis=0).astype('uint8')

fig, ax = plt.subplots()
ax.plot(window_transform)
ax.set_xlabel(r'Input, $f(\mathbf{x})$')
ax.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_aspect('equal')
plt.show()

image_window_transformed = cv.LUT(img, window_transform)
show_img(img, image_window_transformed)

# %% [markdown]
# ## Question 1. (d) Gamma Correction

# %%
# Gamma correction

gamma = 2.
table = np.array([(i/255.0)**(gamma)*255.0 for i in np.arange(0,256)]).astype('uint8')
img_gamma = cv.LUT(img, table)
show_img(img, img_gamma)

# %% [markdown]
# ## Question 1. (e) Gaussian Smoothing

# %%
# Gaussian smoothing

ksize = 11 ; sigma = 4
image_smoothed = cv.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
show_img(img, image_smoothed)

# %% [markdown]
# ## Question 1. (f) Unsharp Masking

# %%
# Unsharp masking

sigma = 1

X = np.arange(-5, 5.25, 1)
Y = np.arange(-5, 5.25, 1)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-(X**2 + Y**2)/(2*sigma**2))
imp = np.zeros(Z.shape)
print(Z.shape)
imp[int(Z.shape[0]/2), int(Z.shape[1]/2)] = 7
Z = imp - Z
unsharp_image = cv.filter2D(img,-1,Z)

show_img(img, unsharp_image )

# %% [markdown]
# ## Question 1. (g) Median Filtering

# %%
# Median filtering

im_noise =  noisy('s&p', img)
im_median = cv.medianBlur(im_noise, 5)
show_img(im_noise,im_median)

# %% [markdown]
# ## Question 1. (h) Bilateral Filtering

# %%
# Bilateral Filtering

blur = cv.bilateralFilter(img,21,75,75)
show_img(img,blur)

# %% [markdown]
# # Question 2. 
# ## Counting Grains of Rice

# %%
# Rice Counting

rice = cv.imread('./a01images/rice.png', cv.IMREAD_GRAYSCALE)

rice1 = cv.medianBlur(rice, 3) #Remove noise using median filtering
edge=128 ; edge2 = 170 #Define the X values to slice the image into pieces
(thresh, rice21) = cv.threshold(rice1[0:edge], 172, 255, cv.THRESH_BINARY)
(thresh, rice22) = cv.threshold(rice1[edge:edge2], 118, 255, cv.THRESH_BINARY)
(thresh, rice23) = cv.threshold(rice1[edge2:256], 114, 255, cv.THRESH_BINARY)
rice2 = np.append(rice21, rice22,axis = 0) #Append the seperately thresholded image parts
rice2 = np.append(rice2, rice23,axis=0)

num_labels, labels	=	cv.connectedComponents(rice2)
print(num_labels)
show_img(rice,rice1,10)

#hist = cv.calcHist([rice], [0], None, [256], [0,256])
#plt.plot(hist)
#plt.xlim([0,256])
#plt.show()

# %% [markdown]
# ## Visualize Using a Color Map

# %%
# Colour map

im_color = cv.applyColorMap(rice2, cv.COLORMAP_WINTER)
show_img(rice2,cv.cvtColor(im_color, cv.COLOR_BGR2RGB),10)

# %% [markdown]
# ## Custom Color Maps For Better Visualization

# %%
# Custom Colour Maps

# Gradient Colour Map
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
labeled_img[label_hue==0] = 0

#Random Colors on Components
transform = np.random.randint(1,256, 255,dtype='uint8')
transform = np.insert(transform, 0, 0)

label_hue = cv.LUT(label_hue, transform)
blank_ch = 255*np.ones_like(label_hue)
labeled_img1 = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img1[label_hue==0] = 0
labeled_img1 = cv.cvtColor(labeled_img1, cv.COLOR_HSV2BGR)

show_img(cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB),labeled_img1, 10)

# %% [markdown]
# # Question 3. Zooming Images by a Given Factor
# ## Calculating Sum of Squared Differences

# %%
def compare_ssd(im1,im2):
    dif = im1.ravel() - im2.ravel()
    return np.dot( dif, dif )


# %%
# Bilinear Interpolation

def bilinear(img,factor):
    height,width,channels = img.shape ; new_height = int(factor*height) ; new_width = int(factor*width)
    zoomed_image=np.zeros((new_height,new_width,channels),np.uint8)
    value=[0,0,0]
 
    for i in range(new_height):
        for j in range(new_width):
            x = i/factor ; y = j/factor
            p=(i+0.0)/factor-x ; q=(j+0.0)/factor-y
            x=int(x)-1 ; y=int(y)-1
            for k in range(3):
                if x+1<new_height and y+1<new_width:
                    value[k]=int(img[x,y][k]*(1-p)*(1-q)+img[x,y+1][k]*q*(1-p)+img[x+1,y][k]*(1-q)*p+img[x+1,y+1][k]*p*q)
            zoomed_image[i, j] = (value[0], value[1], value[2])
    return zoomed_image


# %%
# Nearest Neighbour

def nearest(img,factor):
    height,width,channels = img.shape ; new_height = int(factor*height) ; new_width = int(factor*width)
    zoomed_image=np.zeros((new_height, new_width,channels),np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            x=int(i/factor)
            y=int(j/factor)
            zoomed_image[i,j]=img[x,y]
    return zoomed_image


# %%
# Zoom function

def zoom(im, factor, interpolation):
    if (0<factor<=10):
        if interpolation == 'bilinear':
            return bilinear(im,factor)
        elif interpolation == 'nearest':
            return nearest(im,factor)
    else: print("Invalid factor value")


# %%
# Zoom using opencv Function

def zoom_cv(im, factor, interpolation):
    if (0<factor<=10):
        if interpolation == 'bilinear':
            return cv.resize(im, (im.shape[1]*factor,im.shape[0]*factor), interpolation=cv.INTER_LINEAR)
        elif interpolation == 'nearest':
            return cv.resize(im, (im.shape[1]*factor,im.shape[0]*factor), interpolation= cv.INTER_NEAREST)
    else: print("Invalid factor value")


# %%
# Testing on Images

image_paths = [['./a01images/im01small.png','./a01images/im01.png'],['./a01images/im02small.png','./a01images/im02.png'],['./a01images/im04small.png','./a01images/im04.png']]

for path in image_paths:
    im = cv.imread(path[0], cv.IMREAD_COLOR)
    try :
        im1 = cv.cvtColor(zoom(im,4,'nearest'), cv.COLOR_BGR2RGB)
        im2 = cv.cvtColor(cv.imread(path[1], cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        print(im1.shape,im2.shape)
        print(compare_ssd(im2,im1))
        show_img(im1,im2,16)
    except: print("error occured")


# %%



