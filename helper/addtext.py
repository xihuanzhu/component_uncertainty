#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


# all_dir = "/raid/zhuxihuan/data/body/test2_npy_data"
# train_dir = os.listdir("/raid/zhuxihuan/data/body/two_pre_test2_for_train_dir")
# val_dir = os.listdir("/raid/zhuxihuan/data/body/two_pre_test2_for_val_dir")
# val_test = []
# for s in os.listdir(all_dir):
#     if(s not in train_dir and s not in val_dir):
#         val_test.append(s)
# print(val_test)


# In[3]:


import pickle
import matplotlib.pyplot as plt
import numpy as  np
# read_path = "/home/zhuxihuan/code/MetaSeg/expe2/vval_measure/components/best-loss-037epoch.bin/components257.p"
# component =  pickle.load( open( read_path, "rb" ))
# print(component)
# print(np.max(component), np.min(component))


# In[4]:


# plt.imshow(component)


# In[5]:


from collections import  Counter
def visualize_text_segments( comp, metric ):
    #print(np.unique(comp))
    R = np.asarray( metric )
    R = 1-0.5*R
    G = np.asarray( metric )
    B = 0.3+0.35*np.asarray( metric )
  
    R = np.concatenate( (R, np.asarray([0,1])) )
    G = np.concatenate( (G, np.asarray([0,1])) )
    B = np.concatenate( (B, np.asarray([0,1])) )
    
    #ssum = np.zeros(len(G))
    ssum_x = np.zeros(len(G))
    ssum_y = np.zeros(len(G))
#     for i  range(len(G)):
#         for(x)
  
    components = np.asarray(comp.copy(), dtype='int16')
    components[components  < 0] = len(R)-1
    components[components == 0] = len(R)

    ssum_x = np.zeros(len(G))
    ssum_y = np.zeros(len(G))
    flag = np.zeros(len(G))

    print(np.unique(components))
    #print(flag)
    img = np.zeros( components.shape+(3,) )
    #print("in the visualoze")
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if components[x, y] != len(R)-1  and flag[components[x, y]- 1] == 0:
                # print(np.unique(components))
                # print(components[x, y] - 1)
                flag[components[x, y] - 1] = 1
                ssum_x[components[x, y] - 1] = x
                ssum_y[components[x, y] - 1] = y
            #print(components[x,y]-1)
            #print(flag)
            img[x,y,0] = R[components[x,y]-1]
            img[x,y,1] = G[components[x,y]-1]
            img[x,y,2] = B[components[x,y]-1]
  
    img = np.asarray( 255*img ).astype('uint8')
  
    return img, ssum_x, ssum_y, flag


# In[6]:


# piou = np.loadtxt("/home/zhuxihuan/code/MetaSeg/expe2/train_measure/iou_seg_vis/origin/true/P257_IOU.txt")
# print(piou)


# # In[7]:


# tiou = np.loadtxt("/home/zhuxihuan/code/MetaSeg/expe2/train_measure/iou_seg_vis/origin/true/T257_IOU.txt")
# print(tiou)


# In[18]:


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
def autoAddTextToImage(component, tiou, piou):
    """
    component：图片中每个块的区分数组
    tiou：每个块真实IOU
    piou：每个块回归的IOU
    """
    img, pssum_x, pssum_y, pflag = visualize_text_segments(component, piou)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    imgw, ssum_xw, ssum_yw, flagw = visualize_text_segments(component, tiou)
    imgw = Image.fromarray(imgw.astype('uint8'), 'RGB')
    
    fontpath = "/home/zhuxihuan/code/MetaSeg/helper/simkai.ttf"  # 32为字体大小
    font = ImageFont.truetype(fontpath, 32)
    
    tdraw = ImageDraw.Draw(imgw)
    #print(flag)
#     plt.figure(figsize=(8, 8))
#     plt.clf()
#     plt.subplot(1, 2, 1)
    for s in range(len(pflag) - 2):
        if(pflag[s] == 1):
            #plt.text(ssum_y[s]/512.0 * 500, ssum_x[s]/512.0 * 500 + 20, round(tiou[s], 3))
            #print(type(str(round(tiou[s], 3))))
            tdraw.text((pssum_y[s], pssum_x[s]), str(round(tiou[s], 3)), font = font, fill = (0, 0, 255))
    
#     plt.subplot(1, 2, 2)
    #lt.imshow(img1)
    pdraw = ImageDraw.Draw(img)
    for s in range(len(flagw) - 2):
        if(flagw[s] == 1):
            #plt.text(ssum_y[s]/512.0 * 500, ssum_x[s]/512.0 * 500 + 20, round(piou[s], 3))
            pdraw.text((ssum_yw[s], ssum_xw[s]), str(round(piou[s], 3)), font = font, fill = (0, 0, 255))
            
    f_img = np.concatenate( (imgw,img), axis=1 )
    image = Image.fromarray(f_img.astype('uint8'), 'RGB')
    #image.save("./img.png")
    return image


# In[19]:


#autoAddTextToImage(component, tiou, piou)


# In[ ]:


# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont
# img, ssum_x, ssum_y, flag = visualize_segments(component, piou)
# img = Image.fromarray(img.astype('uint8'), 'RGB')
# imgw, ssum_xw, ssum_yw, flagw = visualize_segments(component, tiou)
# imgw = Image.fromarray(imgw.astype('uint8'), 'RGB')
# imgw.save("imm.png")
# #img1 = img.copy()
# fontpath = "simkai.ttf"  # 32为字体大小
# font = ImageFont.truetype(fontpath, 32)
# draw = ImageDraw.Draw(imgw)
# print(flag)
# plt.figure(figsize=(8, 8))
# plt.clf()
# plt.subplot(1, 2, 1)
# for s in range(len(flag) - 2):
#     if(flag[s] == 1):
#         #plt.text(ssum_y[s]/512.0 * 500, ssum_x[s]/512.0 * 500 + 20, round(tiou[s], 3))
#         #print(type(str(round(tiou[s], 3))))
#         draw.text((ssum_y[s], ssum_x[s]), str(round(tiou[s], 3)), font = font, fill = (0, 0, 255))
# plt.imshow(imgw)
# plt.subplot(1, 2, 2)
# #lt.imshow(img1)
# draw1 = ImageDraw.Draw(img)
# for s in range(len(flag) - 2):
#     if(flag[s] == 1):
#         #plt.text(ssum_y[s]/512.0 * 500, ssum_x[s]/512.0 * 500 + 20, round(piou[s], 3))
#         draw1.text((ssum_y[s], ssum_x[s]), str(round(piou[s], 3)), font = font, fill = (0, 0, 255))
# plt.imshow(img)
# f_img = np.concatenate( (imgw,img), axis=1 )
# image = Image.fromarray(f_img.astype('uint8'), 'RGB')
# image.save("./img.png")


# In[ ]:





# In[ ]:




