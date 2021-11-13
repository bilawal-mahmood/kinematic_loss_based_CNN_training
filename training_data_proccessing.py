from __future__ import print_function, division
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
from glob import glob
import csv
import json
import cv2
import pickle
from mpl_toolkits.mplot3d import axes3d
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import get_affine_transform, affine_transform
from utils.image import transform_preds
import math
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from sklearn import linear_model
#my code
filename = glob(r'D:\NewUnityProject(4)\Labels\*.txt')
data_2D = np.zeros((len(filename),11,2)).astype(int)
for i in range(len(filename)):
    train = pd.read_csv(filename[i])
    data=train.to_numpy()
    b = np.zeros((11,2)).astype(int)
    #b[0,:] = (data[4,1:3].astype(int))
    #b[1,:] = (data[5,1:3].astype(int))
    #b[2,:]=(data[7,1:3].astype(int))
    #b[3,:]=(data[8,1:3].astype(int))
    b=data[0:,1:3].astype(int)
    b[:,1]= 1280-b[:,1]
    # b = (b * [64 / 256, 64 / 256]).astype(int)
    data_2D[i]=b
ext = ".jpg"


image_names=[]
for i in range(len(filename)):
    image_names.append('{0:04}'.format(i))
for i in range(len(filename)):
    image_names[i] = image_names[i] + ext
root_dir = r"D:\NewUnityProject(4)\Screenshots"
filename2 = glob(r'D:\NewUnityProject(4)\Matrix\*.txt')
matrcs = np.ones((len(filename2),4,4)).astype(float)
for i in range(len(filename2)):
    train2 = pd.read_csv(filename2[i])
    data=train2.to_numpy()
    b = np.ones((4,4)).astype(float)
    b[0,:4]=data[0,0:4].astype(float)
    b[1,:4]=data[0,4:8].astype(float)
    b[2,:4]=data[0,8:12].astype(float)
    b[3,:4]=data[0,12:16].astype(float)
    matrcs[i] = b
filename1 = glob(r'D:\NewUnityProject(4)\Position\*.txt')
data_3D_real = np.ones((len(filename1),11,4)).astype(float)
for i in range(len(filename1)):
    train1 = pd.read_csv(filename1[i])
    data=train1.to_numpy()
    b = np.ones((11,4)).astype(float)
    #b[0,:] = (data[4,1:3].astype(int))
    #b[1,:] = (data[5,1:3].astype(int))
    #b[2,:]=(data[7,1:3].astype(int))
    #b[3,:]=(data[8,1:3].astype(int))
    b[:,:3]=data[0:,1:4].astype(float)
    #b[:,1]= 1280-b[:,1]
    # b = (b * [64 / 256, 64 / 256]).astype(int)
    data_3D_real[i]=b
    #print(data_3D_real[i])
#mat=np.zeros((4, 4))
#mat[:][:][:][:] = [[1, 0,0,-1.7],[0,1,0,-3.3],[0,0,-1,-31.9],[0,0,0,1]]
data_3D_trans = np.zeros((len(filename1),11,4)).astype(float)
for i in range(len(filename1)):
    for j in range(11):
        data_3D_trans[i][j]=matrcs[i].dot(data_3D_real[i][j])


gt_3dd = np.zeros((len(filename),11,3)).astype(float)
data_3D = np.zeros((len(filename),11)).astype(float)
for i in range(len(filename)):
    train = pd.read_csv(filename[i])
    data=train.to_numpy()
    b = np.zeros((11,1)).astype(int)
    #b[0,:] = (data[4,1:3].astype(int))
    #b[1,:] = (data[5,1:3].astype(int))
    #b[2,:]=(data[7,1:3].astype(int))
    #b[3,:]=(data[8,1:3].astype(int))
    b=data[0:,3].astype(float)
    #scale, s3d = 0, 0
    #x_values = np.ones(10)
    #y_values = np.ones(10)
    #x_values = data_2D[i,:,0]
    #y_values = data_2D[i,:,1]
    #max_image_x = np.argmax(x_values)
    #min_image_x = np.argmin(x_values)
    #max_image_y = np.argmax(y_values)
    #min_image_y = np.argmin(y_values)
    #dist_x = data_2D[i,max_image_x,0]-data_2D[i,min_image_x,0]
    #dist_y = data_2D[i,max_image_y,1]-data_2D[i,min_image_y,1]
    #if(dist_x>dist_y):
    #    s3d = data_3D_trans[i,max_image_x,0]-data_3D_trans[i,min_image_x,0]
    #    scale = s3d/dist_x
    #else:
    #    s3d = data_3D_trans[i,max_image_y,1]-data_3D_trans[i,min_image_y,1]
    #    scale = s3d/dist_y
    #for j in range(10):
    #    #real_width = 2 * data[j,3] * math.tan(80/2*3.14/180)
    #    #scale_factor = 64/real_width
    #    b[j]=(b[j]*64) / (scale*1280)
    data_3D[i]=b
#for i in range(len(filename)):
    gt_3dd[i,:,0:2] = data_2D[i,:,0:2]
    gt_3dd[i,:,2] = data_3D[i,:]-data_3D[i,0]


class H36M(Dataset):
  def __init__(self, opt, split):
    print('==> initializing 3D {} data.'.format(split))
    self.num_joints = 5
    self.num_eval_joints = 5
    #self.h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
    #self.mpii_to_h36m = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, \
                         #13, 14, 15, 12, 11, 10]
    self.acc_idxs = [0,1,2,3,4]
    self.data_3D_trans=data_3D_trans
    self.image_names= image_names
    self.data_2D = data_2D
    self.gt_3dd = gt_3dd
    self.data_3D = data_3D
    self.root_dir = root_dir
    self.shuffle_ref = []
    self.shuffle_ref_3d = []
    self.edges = [[0,1],[1,2],[2,4],[3,4],[2,3]]
    self.edges_3d = []
    self.mean_bone_length = 4000
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.aspect_ratio = 1.0 * opt.input_w / opt.input_h
    self.split = split
    self.opt = opt
    split_ = split[0].upper() + split[1:]
    #self.image_path =  os.path.join(
    #  self.opt.data_dir, 'h36m', 'ECCV18_Challenge', split_, 'IMG')
    
    

    #ann_path = os.path.join(
    #  self.opt.data_dir, 'h36m', 'msra_cache',
    #  'HM36_eccv_challenge_{}_cache'.format(split_),
    #  'HM36_eccv_challenge_{}_w288xh384_keypoint_jnt_bbox_db.pkl'.format(split_)
    #)
    #self.annot = pickle.load(open(ann_path, 'rb'))
    # dowmsample validation data
    self.idxs = np.arange(len(self.data_2D)) 
    #            else np.arange(0, len(self.annot), 1 if opt.full_test else 10)
    self.num_samples = len(self.data_2D)
    print('Loaded 3D {} {} samples'.format(split, self.num_samples))


  def _load_image(self, index):
      img_name = os.path.join(self.root_dir, self.image_names[index])
      img = cv2.imread(img_name)
      return img
  
  def _get_part_info(self, index):
    #ann = self.annot[self.idxs[index]]
    gt_3d = (gt_3dd[index]).copy().astype(np.float32)
    
    #pts = np.array(ann['joints_3d'], np.float32)[self.h36m_to_mpii]
    pts = np.array(data_2D[index],np.float32)
    # pts[:, :2] = np.array(ann['det_2d'], dtype=np.float32)[:, :2]
    #c = np.array([ann['center_x'], ann['center_y']], dtype=np.float32)
    #s = max(ann['width'], ann['height'])
    c = np.float32([128,128])
    s = [256]
    return gt_3d, pts, c, s
      
  def __getitem__(self, index):
    if index == 0 and self.split == 'train':
      self.idxs = np.random.choice(
        self.num_samples, self.num_samples, replace=False)
    img = self._load_image(index)
    gt_3d, pts, c, s = self._get_part_info(index)
    #print("index",index)
    #print("gt",gt_3d)
    #print("img",img.shape)
    r = 0
    
    #if self.split == 'train':
    #  sf = self.opt.scale
    #  s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    #  # rf = self.opt.rotate
    #  # r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
    #  #    if np.random.random() <= 0.6 else 0

    #flipped = (self.split == 'train' and np.random.random() < self.opt.flip)
    #if flipped:
    #  img = img[:, ::-1, :]
    #  c[0] = img.shape[1] - 1 - c[0]
    #  gt_3d[:, 0] *= -1
    #  pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
    #  for e in self.shuffle_ref_3d:
    #    gt_3d[e[0]], gt_3d[e[1]] = gt_3d[e[1]].copy(), gt_3d[e[0]].copy()
    #  for e in self.shuffle_ref:
    #    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
    
    #s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    s = np.array([s, s])
    #s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)
    new_w_max = np.amax(pts[:,0]).astype(int)
    new_h_max = np.amax(pts[:,1]).astype(int)
    new_w_min = np.amin(pts[:,0]).astype(int)
    new_h_min = np.amin(pts[:,1]).astype(int)
    if(new_h_min-100 <= 0):
        h_minus = 0
    else:
        h_minus=new_h_min-100
    if(new_w_min-100<=0):
        w_minus = 0
    else:
        w_minus = new_w_min-100
    if(new_h_max+100>=1280):
        h_plus = 1280
    else:
        h_plus = new_h_max+100
    if(new_w_max+100>=1280):
        w_plus = 1280
    else:
        w_plus = new_w_max+100
    croped = img[(h_minus):(h_plus) ,(w_minus):(w_plus)]
    pts[:,0] = pts[:,0] - new_w_min+100
    pts[:,1] = pts[:,1] - new_h_min+100
    h= croped.shape[1] 
    w =croped.shape[0]
    seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
            iaa.GammaContrast((0.5, 1.5)),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
            ,iaa.Invert(0.15, per_channel=0.05)
            ,iaa.JpegCompression(compression=(50,99))
            #,iaa.AveragePooling(1,2)
            ])
    #    #inp=cv2.resize(croped, (256,256))
    croped = seq(image=croped)
    desired_size = 256
    old_size = croped.shape[:2]
    croped_scale = max(old_size)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(croped, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [255, 255, 255]
    inp = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
    for i in range(11):
        if pts[i, 0] > 0 or pts[i, 1] > 0:
            pts[i] = pts[i]*[new_size[1]/h,new_size[0]/w]
    pts[:,0] = pts[:,0] + left
    pts[:,1] = pts[:,1] + top
    scale, s3d = 0, 0
    x_values = np.ones(11)
    y_values = np.ones(11)
    x_values = pts[:,0]
    y_values = pts[:,1]
    max_image_x = np.argmax(x_values)
    min_image_x = np.argmin(x_values)
    max_image_y = np.argmax(y_values)
    min_image_y = np.argmin(y_values)
    dist_x = pts[max_image_x,0]-pts[min_image_x,0]
    dist_y = pts[max_image_y,1]-pts[min_image_y,1]
    if(dist_x>dist_y):
        s3d = self.data_3D_trans[index,max_image_x,0]-data_3D_trans[index,min_image_x,0]
        scale = abs(s3d/dist_x)
    else:
        s3d = self.data_3D_trans[index,max_image_y,1]-self.data_3D_trans[index,min_image_y,1]
        scale = abs(s3d/dist_y)
    #trans_input = get_affine_transform(
    #  c, s, r, [self.opt.input_w, self.opt.input_h])
    #inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h),
    #                     flags=cv2.INTER_LINEAR)
    #inp=cv2.resize(img, (256,256))
    #plt.imshow(inp)
    #plt.scatter(pts[:,0],pts[:,1],s=50,marker='.', c='r')
    #plt.pause(0.1) 
    #plt.figure()
    #plt.show()
    inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    gt_3d = gt_3d[0:5]
    b = np.zeros((5,1)).astype(float)
    for j in range(5):
        #real_width = 2 * data[j,3] * math.tan(80/2*3.14/180)
        #scale_factor = 64/real_width
        #b[j]=(gt_3d[j,2]*64) / (scale*croped_scale)
        b[j]=gt_3d[j,2] / scale
    #data_3D[i]=b
    #trans_output = get_affine_transform(
    #  c, s, r, [self.opt.output_w, self.opt.output_h])
    
    #print("points",self.data_3D_trans[index],end='\n')
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                    dtype=np.float32)
    reg_target = np.zeros((self.num_joints, 1), dtype=np.float32)
    reg_ind = np.zeros((self.num_joints), dtype=np.int64)
    reg_mask = np.zeros((self.num_joints), dtype=np.uint8)
    pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
    for i in range(self.num_joints):
        if pts[i, 0] > 0 or pts[i, 1] > 0:
                pts_crop[i] = pts[i]*[self.opt.output_h/256,self.opt.output_h/256]
                out[i] = draw_gaussian(out[i], pts_crop[i], self.opt.hm_gauss)
                reg_target[i] = b[i]/256
                reg_ind[i] = pts_crop[i][1] * self.opt.output_w * self.num_joints + \
                     pts_crop[i][0] * self.num_joints + i # note transposed
    #for i in range(self.num_joints):
      #pt = affine_transform(pts[i, :2], trans_output).astype(np.int32)
      #if pt[0] >= 0 and pt[1] >=0 and pt[0] < self.opt.output_w \
        #and pt[1] < self.opt.output_h:
        #pts_crop[i] = pt
        #out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss)
        #print(pts[i, 2])
        #reg_target[i] = data_3D[i] # assert not self.opt.fit_short_side
        #reg_ind[i] = pt[1] * self.opt.output_w * self.num_joints + \
                     #pt[0] * self.num_joints + i # note transposed
        
        reg_mask[i] = 1
    #print("reg_target",reg_target,end='\n')
    pts_2d_64=pts_crop.copy()
    depth_256=b.copy()
    pts_2d_cen=pts_2d_64[:]-pts_2d_64[0]
    depth_64=depth_256*64/256
    reg1 = linear_model.LinearRegression(fit_intercept=False).fit(pts_2d_cen, depth_64)
    slopes=reg1.coef_.astype(np.float32)
    #print(slopes)
    bck_depth_slp1=((pts_2d_cen[3,0]*slopes[0,0]+pts_2d_cen[3,1]*slopes[0,1])/64).astype(np.float32)
    bck_depth_slp2=((pts_2d_cen[4,0]*slopes[0,0]+pts_2d_cen[4,1]*slopes[0,1])/64).astype(np.float32)
    meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
            'gt_3d': gt_3d, 'pts_crop': pts_crop}
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #x, y, z = np.zeros((3, 5)) 
    #for j in range(5):
     # x[j] = pts_crop[j,0]
     # z[j] = reg_target[j,0]
     # y[j] = pts_crop[j,1]
    #for e in self.edges:
    #  ax.plot(x[e], y[e], z[e], 'gray')
    #plt.show() 
    return {'input': inp, 'target': out, 'meta': meta, 
            'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask, 'slopes':slopes,'depth_64':depth_64,'bck_depth_slp1':bck_depth_slp1,'bck_depth_slp2':bck_depth_slp2}
    
  def __len__(self):
    return self.num_samples


  def convert_eval_format(self, pred):
    pred_h36m = pred
    #pred_h36m[7] = (pred_h36m[0] + pred_h36m[8]) / 2
    #pred_h36m[9] = (pred_h36m[8] + pred_h36m[10]) / 2
    sum_bone_length = self._get_bone_length(pred_h36m)
    mean_bone_length = self.mean_bone_length
    pred_h36m = pred_h36m * mean_bone_length / sum_bone_length
    return pred_h36m

  def _get_bone_length(self, pts):
    sum_bone_length = 0
    #pts = np.concatenate([pts, (pts[14] + pts[11])[np.newaxis, :] / 2])
    for e in self.edges_3d:
      sum_bone_length += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
    return sum_bone_length
