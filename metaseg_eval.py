import math
import random
import os
import time
import sys
from PIL import Image
import numpy as np
import pandas as pd
import scipy
from sklearn import datasets, linear_model, preprocessing, model_selection
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from scipy.interpolate import interp1d
from multiprocessing import Pool
import pickle
from helper.label import trainId2label
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import pickle



# compiled functions for metric calculation
from metrics import compute_metrics
from extract_metrics import computeMetrics

# include io functions and initialize "metaseg"
# NOTE: please check "metaseg_io.py", in particular "probs_gt_save"
# for instructions on how to prepare your input data for MetaSeg.
# Furthermore, please adjust the variables and paths in "global_defs.py"
from metaseg_io import probs_gt_save, probs_gt_load, \
                       metrics_dump, pixel_metrics_dump, metrics_load, \
                       components_dump, pixel_components_dump, components_load, pixel_components_load, \
                       get_save_path_probs_i, \
                       get_save_path_metrics_i, get_save_path_components_i, get_save_path_pixel_components_i, \
                       get_iou_seg_vis_path_i, get_save_path_stats, \
                       get_img_path_fname, metaseg
                     
from metaseg_plot import add_scatterplot_vs_iou, make_scatterplots, \
                         plot_roc_curve, name_to_latex, generate_lasso_plots, \
                         plot_regression

# NOTE: 
# "cs_labels" is included for the segmentations color code, this is only required for visualization.
# Replace this if necessary and modify the lines in "visualize_metrics_i()" that contain "cs_labels"
# accordingly.
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers
import torch
import torch.nn as nn
import torch.nn.functional as F

import keras
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import os
from helper.addtext import autoAddTextToImage
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed( 0 )

INDEX = 10

def main():
  
  metaseg.set_from_argv( sys.argv )
  
  metaseg.print_attr()
  
  if metaseg.get("COMPUTE_METRICS"):
    compute_metrics_per_image()
  
  if metaseg.get("VISUALIZE_METRICS"):
    visualize_metrics()
  
  if metaseg.get("ANALYZE_METRICS"):
    analyze_metrics()

def get_nn_model(X_train):
    reg = regularizers.l2(5e-9)
    reg1 = regularizers.l1(1e-9)
    
    model = Sequential()
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg, input_dim=X_train.shape[-1]))
    model.add(Dense(units=61, activation='relu' ))
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg ))
    #model.add(BatchNormalization())
    #model.add(Dense(units=61, activation='relu' ))
    model.add(Dense(units=31, activation='relu', kernel_regularizer=reg ))
    model.add(Dense(units=1, kernel_regularizer=reg, activation='linear'))
    
    def stddev(y_true, y_pred):
      return K.sqrt( keras.losses.mean_squared_error(y_true, y_pred) )
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[stddev])
    return model

def label_as_onehot(label, num_classes, shift_range=0):
  
  y = np.zeros((num_classes, label.shape[0], label.shape[1]))
  for c in range(shift_range,num_classes+shift_range):
          y[c-shift_range][label==c] = 1
  y = np.transpose(y,(1,2,0)) # shape is (height, width, num_classes)
  return y.astype('uint8')



def classes_to_categorical( classes, nc = None ):

  classes = np.squeeze( np.asarray(classes) )
  if nc == None:
    nc      = np.max(classes)
  classes = label_as_onehot( classes.reshape( (classes.shape[0],1) ), nc ).reshape( (classes.shape[0], nc) )
  names   = [ "C_"+str(i) for i in range(nc) ]
  
  return classes, names



def visualize_segments( comp, metric ):
  
  R = np.asarray( metric )
  R = 1-0.5*R
  G = np.asarray( metric )
  B = 0.3+0.35*np.asarray( metric )
  
  R = np.concatenate( (R, np.asarray([0,1])) )
  G = np.concatenate( (G, np.asarray([0,1])) )
  B = np.concatenate( (B, np.asarray([0,1])) )
  
  components = np.asarray(comp.copy(), dtype='int16')
  components[components  < 0] = len(R)-1
  components[components == 0] = len(R)
  
  img = np.zeros( components.shape+(3,) )
  #print("in the visualoze")
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      img[x,y,0] = R[components[x,y]-1]
      img[x,y,1] = G[components[x,y]-1]
      img[x,y,2] = B[components[x,y]-1]
  
  img = np.asarray( 255*img ).astype('uint8')
  
  return img

def modify_visualize_segments( comp, metric, isTrue=False ):
  
  R = np.asarray( metric )
  R = 1-0.5*R
  G = np.asarray( metric )
  B = 0.3+0.35*np.asarray( metric )
  
  R = np.concatenate( (R, np.asarray([0,1])) )
  G = np.concatenate( (G, np.asarray([0,1])) )
  B = np.concatenate( (B, np.asarray([0,1])) )
  
  components = np.asarray(comp.copy(), dtype='int16')
  components[components  < 0] = len(R)-1
  components[components == 0] = len(R)

  img = np.zeros( components.shape+(3,) )
  #print("in the visualoze")
  for i in range(1, np.max(components)):
    ig = np.zeros( components.shape+(3,) )
    for x in range(img.shape[0]):
      for y in range(img.shape[1]):
        if components[x, y] == i: 
          if i == 1:
            print(i, "PPPPPPPPPPPPPPPPPPPP ", components[x, y])
          ig[x,y,0] = R[components[x,y]-1]
          ig[x,y,1] = G[components[x,y]-1]
          ig[x,y,2] = B[components[x,y]-1]
    print(np.max(ig), "*******")
    k = Image.fromarray(np.asarray( 255*ig ).astype('uint8'), 'RGB')
    seg_dir = metaseg.get("IOU_SEG_VIS_DIR")
    if not os.path.exists( seg_dir + "true/"):
      os.makedirs( seg_dir + "true/")
    k.save(seg_dir+"true/imgttttt"+str(i)+".png")


def metrics_to_nparray( metrics, names, normalize=False, non_empty=False, all_metrics=[] ):
  
  #normalize = True
  #print("mmmmm" , np.max(metrics['cprob0']), np.max(metrics['cprob1']))
  I = range(len(metrics['S_in']))
  if non_empty == True:
    I = np.asarray(metrics['S_in']) > 0
  M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
  MM = []
  if all_metrics == []:
    MM = M.copy()
  else:
    MM = np.asarray( [ np.asarray(all_metrics[ m ])[I] for m in names ] )
  
  if normalize == True:
    for i in range(M.shape[0]):
      if names[i] != "class":
        M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
  
  M = np.squeeze(M.T)
  # print(M.shape)
  # print("KKKKKK" , np.max(M[:, -2]), np.max(M[:, -1]))
  return M



def compute_metrics_i( i ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    start = time.time()
    
    probs, gt, _ = probs_gt_load( i )
    metrics, components = compute_metrics( probs, gt )
    
    metrics_dump( metrics, i )
    components_dump( components, i )
    
    print("image", i, "processed in {}s\r".format( round(time.time()-start) ) )



def visualize_metrics_i( iou, iou_pred, i ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    probs, gt, filename = probs_gt_load( i )
    
    path = get_img_path_fname( filename )
    print(path)
    input_image = np.load(path)
    components  = components_load( i )
    
    pred = np.asarray( np.argmax( probs, axis=-1 ), dtype='int' )
    
    gt[ gt == 255 ] = 0
    predc = np.asarray([ trainId2label[ pred[p,q] ].color for p in range(pred.shape[0]) for q in range(pred.shape[1]) ])
    gtc   = np.asarray([ trainId2label[ gt[p,q]   ].color for p in range(gt.shape[0]) for q in range(gt.shape[1]) ])
    input_image = np.array([input_image, input_image, input_image]).transpose(1, 2, 0)
    predc = predc.reshape(input_image.shape)
    gtc   = gtc.reshape(input_image.shape)
    
    img_iou = visualize_segments( components, iou )
    
    I4 = predc / 2.0 + input_image / 2.0 
    I3 = gtc / 2.0 + input_image / 2.0
    
    img_pred = visualize_segments( components, iou_pred )

    img = np.concatenate( (img_iou,img_pred), axis=1 )
    img2 = np.concatenate( (I3,I4), axis=1 )
    img = np.concatenate( (img,img2), axis=0 )
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    
    seg_dir = metaseg.get("IOU_SEG_VIS_DIR")
    if not os.path.exists( seg_dir ):
      os.makedirs( seg_dir )
    image.save(seg_dir+"img"+str(i)+".png")
    
    print("stored:",seg_dir+"img"+str(i)+".png")

    
    
def visualize_metrics( ):
  
  num_cores = metaseg.get("NUM_CORES")
  
  print("visualization running")
  
  metrics = metrics_load( 0 )
  
  start = list([ 0, len(metrics["S"]) ])
  
  for i in range(1,metaseg.get("NUM_IMAGES")):    
    m = metrics_load( i )
    start += [ start[-1]+len(m["S"]) ]
    
    for j in metrics:
      metrics[j] += m[j]
  
  nclasses = np.max(metrics["class"])+1
  
  Xa, classes, ya, _, X_names, class_names = metrics_to_dataset( metrics, nclasses, non_empty=False )
  Xa = np.concatenate( (Xa,classes), axis=-1 )
  X_names += class_names
  
  lmr = linear_model.LinearRegression()
  lmr.fit(Xa,ya)
  
  ya_pred = np.clip( lmr.predict(Xa), 0, 1 )
  
  print("model r2 score:", r2_score(ya,ya_pred) )
  print(" ")
  
  p = Pool(num_cores)
  
  p_args = [ (ya[start[i]:start[i+1]], ya_pred[start[i]:start[i+1]], i) for i in range(10) ]
  
  p.starmap( visualize_metrics_i, p_args )



def concatenate_metrics( train=True, save=False ):
  
  metrics = metrics_load( 0 , train)

  if train:
    num = metaseg.get("NUM_IMAGES")
  else:
    #expe5:2530
    #expe6:train 1800 val 194
    num = 194

  for i in range(1, num):
    sys.stdout.write("\t concatenated file number {} / {}\r".format(i+1, num))
    
    m = metrics_load( i ,train)
    
    for j in metrics:
      metrics[j] += m[j]
  
  print(" ")
  print("connected components:", len(metrics['iou']) )
  print("non-empty connected components:", np.sum( np.asarray(metrics['S_in']) != 0) )
  
  if ( save == True ):
    metrics_dump( metrics, "_all" )
  
  return metrics

def prediction(probs, gt, ignore=True ):
    pred = np.asarray( np.argmax( probs, axis=-1 ), dtype="uint8" )
    if ignore == True:
        pred[ gt==255 ] = 255
    return pred

def compute_pixel_metrics_i( i ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    start = time.time()
    
    compo = components_load( i , train=True)
    probs, gt, _ = probs_gt_load( i )
    seg = np.asarray( prediction(probs, gt, ignore=True ), dtype="uint8" )
    
    components, metrics, block = computeMetrics(gt, probs, seg, compo)
    
    pixel_metrics_dump( metrics, i )
    pixel_components_dump( components, i )
    
    print("image", i, "processed in {}s\r".format( round(time.time()-start) ) )

def compute_metrics_per_image( ):
  
  num_cores = metaseg.get("NUM_CORES")
  
  print("calculating statistics")
      
  p = Pool(num_cores)
  
  p_args = [ (k,) for k in range(metaseg.get("NUM_IMAGES")) ]
  
  p.starmap( compute_pixel_metrics_i, p_args )
  
  concatenate_metrics( save=True )

        

def adjusted_r2(r2, num_dof, num_samples):
  
  return 1 - (1-r2) * (num_samples - 1) / (num_samples - num_dof - 1)

def modify_visualize_metrics_i( iou, iou_pred, i ):
  seg_dir = metaseg.get("IOU_SEG_VIS_DIR")
  if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)
  if os.path.isfile( get_save_path_probs_i(i) ):

    probs = np.load(f"/raid/zhuxihuan/data/body/expe6/val_data/probs/{i}.npy")
    gt  = np.load(f"/raid/zhuxihuan/data/body/expe6/val_data/GT/{i}.npy")
    input_image = np.load(f"/raid/zhuxihuan/data/body/expe6/val_data/IMG/{i}.npy")
    components = pickle.load( open( f"/home/zhuxihuan/code/MetaSeg/expe6/val_measure/components/best-loss-037epoch.bin/components{i}.p", "rb" ) )
    #print("*******npmax: ", np.max(components), "npmin: ", np.min(components))
    pred = np.asarray( np.argmax( probs, axis=-1 ), dtype='int' )
    
    
    gt[ gt == 255 ] = 0
    gt = np.array(gt, dtype='int')
    predc = np.asarray([ trainId2label[ pred[p,q] ].color for p in range(pred.shape[0]) for q in range(pred.shape[1]) ])
    gtc   = np.asarray([ trainId2label[ gt[p,q]   ].color for p in range(gt.shape[0]) for q in range(gt.shape[1]) ])
    input_image = np.array([input_image, input_image, input_image]).transpose(1, 2, 0)
    predc = predc.reshape(input_image.shape)
    gtc   = gtc.reshape(input_image.shape)
    gt_pred = np.concatenate( (gtc, predc), axis=1 )
    gtt = gt
    gtt[gtt == 1] = 255
    predd = pred
    predd[predd == 1] = 255
    gt_bpred = np.concatenate( (gtt, predd), axis=1 )
    #gt_pred = Image.fromarray(gt_pred.astype('uint8'), 'RGB')
    gt_bpred = Image.fromarray(gt_bpred.astype('uint8'))
    if not os.path.exists( seg_dir + "less95_GT/"):
      os.makedirs( seg_dir + "less95_GT/")
    gt_bpred.save(seg_dir+"less95_GT/gt_pred"+str(i)+".jpg")
    #gt_pred.save(seg_dir+"gt_pred"+str(i)+".png")
    
    # img_iou = visualize_segments( components, iou )
    # if False:
    #   modify_visualize_segments(components, iou, True)
    
    # I4 = predc / 2.0 + input_image / 2.0 
    # I3 = gtc / 2.0 + input_image / 2.0
    
    # img_pred = visualize_segments( components, iou_pred )
    # if False:
    #   modify_visualize_segments(components, iou, False)

    # img = np.concatenate( (img_iou,img_pred), axis=1 )
    # img2 = np.concatenate( (I3,I4), axis=1 )
    # img = np.concatenate( (img,img2), axis=0 )
    # image = Image.fromarray(img.astype('uint8'), 'RGB')
    if np.sum(components) != 262144:
      print("are you kill me")
      image = autoAddTextToImage(components, iou, iou_pred)
    
      if not os.path.exists( seg_dir + "less95_IOU/"):
        os.makedirs( seg_dir + "less95_IOU/")
      image.save(seg_dir+"less95_IOU/img"+str(i)+".png")
    
      print("stored:",seg_dir+"img"+str(i)+".png")

def fit_model_run( Xa, RXa, ya, y0a, v_Xa, Rv_Xa, v_ya, v_y0a, alphas, X_names, stats, run ):
  
  if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

  print("run",run)
  
  np.random.seed( run )
  ss = Normalizer()
  n_Xa = ss.fit_transform(Xa)
  n_v_Xa = ss.fit_transform(v_Xa)
  Rn_Xa = ss.fit_transform(RXa)
  Rn_v_Xa = ss.fit_transform(Rv_Xa)
  # n_Xa = Xa
  # n_v_Xa = v_Xa
  # Rn_Xa = RXa
  # Rn_v_Xa = Rv_Xa
  # kk = ss.fit(Xa) 
  # ff = ss.fit(v_Xa)
  # n_Xa = kk.transform(Xa)
  # n_v_Xa = ff.transform(v_Xa)
  # val_mask = np.random.rand(len(ya)) < 3.0/6.0
  
  # Xa_val = Xa[val_mask]
  # ya_val = ya[val_mask]
  # y0a_val = y0a[val_mask]
      
  # Xa_train = Xa[np.logical_not(val_mask)]
  # ya_train = ya[np.logical_not(val_mask)]
  # y0a_train = y0a[np.logical_not(val_mask)]
  # v_Xa = (v_Xa - np.min(v_Xa))/(np.max(v_Xa) - np.min(v_Xa))
  # Xa = (Xa - np.min(Xa))/(np.max(Xa) - np.min(Xa))
  Xa_val = n_v_Xa
  ya_val = v_ya
  y0a_val = v_y0a

  Xa_train = n_Xa
  ya_train = ya
  y0a_train = y0a

  coefs = np.zeros((len(alphas),Xa.shape[1]))
  
  max_acc = 0
  best_lm = []
  
  for i in range(len(alphas)):
  #for i in range(2):
    lm = linear_model.LogisticRegression(C=alphas[i], penalty='l1', solver='saga', max_iter=100000, tol=1e-3 )#, class_weight='balanced')
    lm.fit( Xa_train, y0a_train )
    
    stats['penalized_val_acc'][run,i] = lm.score( Xa_val, y0a_val )
    stats['penalized_train_acc'][run,i] = lm.score( Xa_train, y0a_train )
    
    if stats['penalized_val_acc'][run,i] > max_acc:
      max_acc = stats['penalized_val_acc'][run,i]
      best_lm = lm
    
    print("step"+str(i)+", alpha={:.2E}".format(alphas[i])+", val. acc.: {:.2f}%".format(100*stats['penalized_val_acc'][run,i]), end=", ")
    print("coefs non-zero:", end=" ")
    
    metapr = lm.predict_proba(Xa_val)
    fpr, tpr, _ = roc_curve(y0a_val, metapr[:,1])
    stats['penalized_val_auroc'][run,i] = auc(fpr, tpr)
    
    metapr_t = lm.predict_proba(Xa_train)
    fpr, tpr, _ = roc_curve(y0a_train, metapr_t[:,1])
    stats['penalized_train_auroc'][run,i] = auc(fpr, tpr)
    
    coefs[i] = np.asarray(lm.coef_[0])
    print([ j for j in range(len(coefs[i])) if np.abs(coefs[i,j]) > 1e-6 ])
    
    if np.sum( np.abs(coefs[i]) > 1e-6 ) > 0 :
      lm2 = linear_model.LogisticRegression(penalty='none', solver='saga', max_iter=100000, tol=1e-3 )#, class_weight='balanced')
      lm2.fit( Xa_train[ :, np.abs(coefs[i]) > 1e-6 ], y0a_train )
      
      stats['plain_val_acc'][run,i] = lm2.score( Xa_val[ :, np.abs(coefs[i]) > 1e-6 ], y0a_val )
      stats['plain_train_acc'][run,i] = lm2.score( Xa_train[ :, np.abs(coefs[i]) > 1e-6 ], y0a_train )
      
      metapr = lm2.predict_proba(Xa_val[ :, np.abs(coefs[i]) > 1e-6 ])
      fpr, tpr, _ = roc_curve(y0a_val, metapr[:,1])
      stats['plain_val_auroc'][run,i] = auc(fpr, tpr)
      
      
      metapr_t = lm2.predict_proba(Xa_train[ :, np.abs(coefs[i]) > 1e-6 ])
      fpr, tpr, _ = roc_curve(y0a_train, metapr_t[:,1])
      stats['plain_train_auroc'][run,i] = auc(fpr, tpr)
    else:
      stats['plain_val_acc'][run,i] = stats['penalized_val_acc'][run,i]
      stats['plain_train_acc'][run,i] = stats['penalized_train_acc'][run,i]
      
      stats['plain_val_auroc'][run,i] = stats['penalized_val_auroc'][run,i]
      stats['plain_train_auroc'][run,i] = stats['penalized_train_auroc'][run,i]
      
  
  max_acc = np.argmax(stats['penalized_val_acc'][run])
  
  ypred = best_lm.predict(Xa_val)
  ypred_t = best_lm.predict(Xa_train)
  
  E_ind = 0
  for E_ind in range(len(X_names)):
    if X_names[E_ind] == "E":
      break
  
  lme = linear_model.LogisticRegression(penalty='none', solver='saga')
  lme.fit( Xa_train[:,E_ind].reshape((Xa_train.shape[0],1)), y0a_train )
  
  stats['entropy_val_acc'][run] = lme.score( Xa_val[:,E_ind].reshape((Xa_val.shape[0],1)), y0a_val )
  stats['entropy_train_acc'][run] = lme.score( Xa_train[:,E_ind].reshape((Xa_train.shape[0],1)), y0a_train )
  
  metapr = lme.predict_proba(Xa_val[:,E_ind].reshape((Xa_val.shape[0],1)))
  fpr, tpr, _ = roc_curve(y0a_val, metapr[:,1])
  stats['entropy_val_auroc'][run] = auc(fpr, tpr)

  metapr = lme.predict_proba(Xa_train[:,E_ind].reshape((Xa_train.shape[0],1)))
  fpr, tpr, _ = roc_curve(y0a_train, metapr[:,1])
  stats['entropy_train_auroc'][run] = auc(fpr, tpr)
  
  if run == 0:
    metapr = best_lm.predict_proba(Xa_val)
    plot_roc_curve(y0a_val, metapr[:,1], metaseg.get("RESULTS_DIR")+'roccurve.pdf')
  
  stats['iou0_found'][run] = np.sum( np.logical_and(ypred == 1, y0a_val == 1) ) + np.sum( np.logical_and(ypred_t == 1, y0a_train == 1) )
  stats['iou0_not_found'][run] = np.sum( np.logical_and(ypred == 0, y0a_val == 1) ) + np.sum( np.logical_and(ypred_t == 0, y0a_train == 1) )
  stats['not_iou0_found'][run] = np.sum( np.logical_and(ypred == 0, y0a_val == 0) ) + np.sum( np.logical_and(ypred_t == 0, y0a_train == 0) )
  stats['not_iou0_not_found'][run] = np.sum( np.logical_and(ypred == 1, y0a_val == 0) ) + np.sum( np.logical_and(ypred_t == 1, y0a_train == 0) )
  
  
  X2_train = Rn_Xa
  y2_train = ya_train
  X2_val   = Rn_v_Xa
  y2_val   = ya_val
  #lmr = linear_model.Ridge(max_iter=500)
  lmr = get_nn_model(X2_train)
  lmr.fit(X2_train, y2_train, epochs=100, batch_size=128, verbose=0)
  # lmr = linear_model.LinearRegression()
  #lmr.fit(X2_train,y2_train)
  y2_pred = lmr.predict(X2_val)
  y2_pred_t = lmr.predict(X2_train)
  
  stats['regr_val_mse'][run] = np.sqrt( mean_squared_error(y2_val, y2_pred) )
  stats['regr_val_r2'][run]  = r2_score(y2_val, y2_pred)
  stats['regr_train_mse'][run] = np.sqrt( mean_squared_error(y2_train, y2_pred_t) )
  stats['regr_train_r2'][run]  = r2_score(y2_train, y2_pred_t)
  
  # plt.clf()
  # plt.xlabel(f"tIOU")
  # plt.ylabel("pIOU")
  # #print(np.min(Xa[:, i]), "*********", X_names[i])
  # cmap=plt.get_cmap('tab20')
  # plt.scatter(y2_val, np.clip(y2_pred, 0 ,1), s=5, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
  # # if i == "class":
  # # print(metrics[i][:1000])
  # plt.savefig(f'{metaseg.get("IOU_SEG_VIS_DIR")}true/{run}mm.png')
  t = []
  p = []
  if run == 0 and False:
    #less95 = np.loadtxt("/raid/zhuxihuan/data/body/expe5/val_data/less95.txt")
    #for k in range(2530):
    for k in range(194):
      k = int(k)
      #k = k + 257#78 + 138 + 38
      tmetrics   = pickle.load( open(f"/home/zhuxihuan/code/MetaSeg/expe6/val_measure/metrics/best-loss-037epoch.bin/metrics{k}.p", "rb" ) )
      print(tmetrics)
      #print(len(tmetrics['iou']), "&&&&&&&")
      tnclasses  = 2#np.max( tmetrics["class"] ) + 1
  
      tXa, RtXa, tclasses, Rtclasses, tya, ty0a, tX_names, tclass_names = metrics_to_dataset( tmetrics, tnclasses)

     # Rtclasses = np.array([Rtclasses])
      if np.max( tmetrics["class"] ) == 0:
        tXa = np.array([np.concatenate( (RtXa,Rtclasses), axis=-1 )])
      else:
        tXa = np.array(np.concatenate( (RtXa,Rtclasses), axis=-1 ))
      tX_names += tclass_names
      tXa,  tX_names = select_single_feature(tXa,  tX_names)
      tXa = ss.fit_transform(tXa)
      t_pred = lmr.predict(tXa[:, :])
      t_pred = np.squeeze(t_pred)
      print(" in fit model run ", np.max(t_pred), "***", np.min(t_pred))
      modify_visualize_metrics_i(tya, np.clip(t_pred, 0 ,1), k)
      #modify_visualize_metrics_i(tya, (t_pred - np.min(t_pred))/ (np.max(t_pred) - np.min(t_pred)), k)
      seg_dir = metaseg.get("IOU_SEG_VIS_DIR")
      if not os.path.exists( seg_dir + "true/"):
        os.makedirs( seg_dir + "true/")
      np.savetxt(f"{seg_dir}true/T{k}_IOU.txt", np.array([tya]))
      np.savetxt(f"{seg_dir}true/P{k}_IOU.txt", np.array([np.clip(t_pred, 0 ,1)]))
      #print("True IOU: ", tya)
      #print("Pred IOU: ",np.clip(t_pred, 0 ,1))
      # t = np.hstack([t, tya])
      # p = np.hstack([p, np.clip(t_pred, 0 ,1)])
    # #print(t.shape, "&&&&&")
    # plt.clf()
    # plt.xlabel(f"tIOU")
    # plt.ylabel("pIOU")
    # #print(np.min(Xa[:, i]), "*********", X_names[i])
    # cmap=plt.get_cmap('tab20')
    # plt.scatter(t, p, s=5, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
    # # if i == "class":
    # # print(metrics[i][:1000])
    # plt.savefig(f'{seg_dir}true/{666666}.png')
  
  lmer = linear_model.Ridge(max_iter=500)
  #lmer = linear_model.LinearRegression()
  lmer.fit(X2_train[:,E_ind].reshape((X2_train.shape[0],1)),y2_train)
  y2e_pred = lmer.predict(X2_val[:,E_ind].reshape((X2_val.shape[0],1)))
  y2e_pred_t = lmer.predict(X2_train[:,E_ind].reshape((X2_train.shape[0],1)))

  stats['entropy_regr_val_mse'][run] = np.sqrt( mean_squared_error(y2_val, y2e_pred) )
  stats['entropy_regr_val_r2'][run]  = r2_score(y2_val, y2e_pred)
  stats['entropy_regr_train_mse'][run] = np.sqrt( mean_squared_error(y2_train, y2e_pred_t) )
  stats['entropy_regr_train_r2'][run]  = r2_score(y2_train, y2e_pred_t)
  
  stats['coefs'][run] = np.asarray(coefs)
  print("******************************", np.max(y2_pred), np.min(y2_pred))
  if run == 0:
    print(np.max(ypred), "in fit model run")
    plot_regression( X2_val, y2_val, y2_pred, np.array(ya_val), np.array(ypred), X_names )
    #pass
  return stats



def compute_correlations( metrics ):
  
  pd.options.display.float_format = '{:,.5f}'.format

  df_full = pd.DataFrame( data=metrics )
  df_full = df_full.copy().drop(["class","iou0"], axis=1)
  features = df_full.copy().drop(["iou"], axis=1).columns    
  df_all  = df_full.copy()
  df_full = df_full.copy().loc[df_full['S_in'].nonzero()[0]]
  make_scatterplots("../", df_full, df_full )
  iou_corrs   = df_full.corr()["iou"]
  print("\n\ncorrelations with iou (only non_empty in)")
  print(iou_corrs)
  
  y0a = metrics_to_nparray( metrics, ["iou0"] , normalize=False, non_empty=True )
  print(" ")
  print("IoU=0:", np.sum(y0a==1), "of", y0a.shape[0] )
  print("IoU>0:", np.sum(y0a==0), "of", y0a.shape[0] )
  
  return iou_corrs



def get_alphas( n_steps, min_pow, max_pow ):
  
  m = interp1d([0,n_steps-1],[min_pow,max_pow])
  alphas = [10 ** m(i).item() for i in range(n_steps)]
  
  return alphas



def init_stats( n_av, alphas, X_names ):

  n_steps   = len(alphas)
  n_metrics = len(X_names)# - (17 - INDEX)
  stats     = dict({})
  
  per_alphas_av_stats = ['penalized_val_acc','penalized_val_auroc','penalized_train_acc','penalized_train_auroc', \
                          'plain_val_acc','plain_val_auroc','plain_train_acc','plain_train_auroc', 'coefs' ]
  per_av_stats        = ['entropy_val_acc','entropy_val_auroc','entropy_train_acc','entropy_train_auroc', \
                          'regr_val_mse', 'regr_val_r2', 'regr_train_mse', 'regr_train_r2', \
                          'entropy_regr_val_mse', 'entropy_regr_val_r2', 'entropy_regr_train_mse', 'entropy_regr_train_r2', \
                          'iou0_found', 'iou0_not_found', 'not_iou0_found', 'not_iou0_not_found' ]
                    
  for s in per_alphas_av_stats:
    stats[s] = 0.5*np.ones((n_av,n_steps))
    
  for s in per_av_stats:
    stats[s] = np.zeros((n_av,))
    
  stats["coefs"]        = np.zeros((n_av,n_steps,n_metrics))
  stats["alphas"]       = alphas
  stats["n_av"]         = n_av
  stats["n_metrics"]    = n_metrics
  stats["metric_names"] = X_names
  
  return stats



def merge_stats( stats, single_run_stats, n_av ):
  
  for run in range(n_av):
    for s in stats:
      if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
        stats[s][run] = single_run_stats[run][s][run]
  
  return stats



def dump_stats( stats, metrics ):
  
  iou_corrs = compute_correlations( metrics )
  y0a = metrics_to_nparray( metrics, ["iou0"] , normalize=False, non_empty=True )

  mean_stats = dict({})
  std_stats = dict({})
  
  for s in stats:
    if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
      mean_stats[s] = np.mean(stats[s], axis=0)
      std_stats[s]  = np.std( stats[s], axis=0)
  
  best_pen_ind = np.argmax(mean_stats['penalized_val_acc'])
  best_plain_ind = np.argmax(mean_stats['plain_val_acc'])
  
  # dump stats latex ready
  with open(metaseg.get("RESULTS_DIR")+'av_results.txt', 'wt') as f:
    
    print( iou_corrs, file=f )
    print(" ", file=f )
    
    print("classification", file=f )
    print( "                             & train                &  val                 &    \\\\ ", file= f)
    M = sorted([ s for s in mean_stats if 'penalized' in s and 'acc' in s ])
    print( "ACC penalized               ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'plain' in s and 'acc' in s ])
    print( "ACC unpenalized             ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'entropy' in s and 'acc' in s ])
    print( "ACC entropy baseline        ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    
    M = sorted([ s for s in mean_stats if 'penalized' in s and 'auroc' in s ])
    print( "AUROC penalized             ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'plain' in s and 'auroc' in s ])
    print( "AUROC unpenalized           ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'entropy' in s and 'auroc' in s ])
    print( "AUROC entropy baseline      ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    
    print(" ", file=f)
    print("regression", file=f)
    
    M = sorted([ s for s in mean_stats if 'regr' in s and 'mse' in s and 'entropy' not in s ])
    print( "$\sigma$, all metrics       ", end=" & ", file= f )
    for s in M: print( "${:.3f}".format(mean_stats[s])+"(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'regr' in s and 'mse' in s and 'entropy' in s ])
    print( "$\sigma$, entropy baseline  ", end=" & ", file= f )
    for s in M: print( "${:.3f}".format(mean_stats[s])+"(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f )
    print("   \\\\ ", file=f )
    
    M = sorted([ s for s in mean_stats if 'regr' in s and 'r2' in s and 'entropy' not in s ])
    print( "$R^2$, all metrics          ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'regr' in s and 'r2' in s and 'entropy' in s ])
    print( "$R^2$, entropy baseline     ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    
    print(" ", file=f )
    M = sorted([ s for s in mean_stats if 'iou' in s ])          
    for s in M: print( s, ": {:.0f}".format(mean_stats[s])+"($\pm${:.0f})".format(std_stats[s]), file=f )
    print("IoU=0:", np.sum(y0a==1), "of", y0a.shape[0], "non-empty components", file=f )
    print("IoU>0:", np.sum(y0a==0), "of", y0a.shape[0], "non-empty components", file=f )
    print("total number of components: ", len(metrics['S']), file=f )
    print(" ", file=f )
    
    dump_path = get_save_path_stats()
    dump_dir  = os.path.dirname( dump_path )

    if not os.path.exists( dump_dir ):
      os.makedirs( dump_dir )

    pickle.dump( stats, open( dump_path, "wb" ) )
    

  return mean_stats, std_stats



def metrics_to_dataset( metrics, nclasses, non_empty=True, all_metrics=[] ):
  
  class_names = []
  X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0"] and "cprob" not in m ])
  
  if metaseg.get("CLASS_DTYPE") == metaseg.get("CLASS_DTYPES")[1]:
    class_names = [ "cprob"+str(i) for i in range(nclasses) if "cprob"+str(i) in metrics ]
  elif metaseg.get("CLASS_DTYPE") == metaseg.get("CLASS_DTYPES")[0]:
    class_names = ["class"]
  
  print(class_names, "&&&&^")
  
  #print("in metrics", len(metrics['iou']))
  CXa     = metrics_to_nparray( metrics, X_names    , normalize=False , non_empty=non_empty, all_metrics=all_metrics )
  RXa     = metrics_to_nparray( metrics, X_names    , normalize=False , non_empty=non_empty, all_metrics=all_metrics )
  Cclasses = metrics_to_nparray( metrics, class_names, normalize=False , non_empty=non_empty, all_metrics=all_metrics )
  Rclasses = metrics_to_nparray( metrics, class_names, normalize=False , non_empty=non_empty, all_metrics=all_metrics )
  ya      = metrics_to_nparray( metrics, ["iou" ]   , normalize=False, non_empty=non_empty )
  y0a     = metrics_to_nparray( metrics, ["iou0"]   , normalize=False, non_empty=non_empty )
  
  #print("in metrics", Xa.shape)
  if metaseg.get("CLASS_DTYPE") == metaseg.get("CLASS_DTYPES")[0]:
    classes, class_names = classes_to_categorical( classes, nclasses )
  
  print(Rclasses, "*&*&%%^&%^&")
  return CXa, RXa, Cclasses, Rclasses, ya, y0a, X_names, class_names

def select_single_feature(Xa,  X_names):
  start = [0, 5, 14, 16]
  end = [2, 7, 15, 17]
  #print(start[0].type)
  tXa = np.hstack((Xa[:, start[0]:end[0]], Xa[:, start[1]:end[1]]))
  tX_names = np.hstack((X_names[start[0]:end[0]], X_names[start[1]:end[1]]))
  for s in range(2, len(start)):
    tXa = np.hstack((tXa[:, :], Xa[:, start[s]:end[s]]))
    tX_names = np.hstack((tX_names[:], X_names[start[s]:end[s]]))
  # tXa = Xa[:, start[0]:end[0]]
  # tRXa = RXa[:, start[0]:end[0]]
  # tv_Xa = v_Xa[:, start[0]:end[0]]
  # tRv_Xa = Rv_Xa[:, start[0]:end[0]]
  # tX_names = X_names[start[0]:end[0]]
  return tXa, tX_names

def select_feature(Xa, RXa, v_Xa, Rv_Xa, X_names):
  start = [0, 5, 14, 16]
  end = [2, 7, 15, 17]
  #print(start[0].type)
  tXa = np.hstack((Xa[:, start[0]:end[0]], Xa[:, start[1]:end[1]]))
  tRXa = np.hstack((RXa[:, start[0]:end[0]], RXa[:, start[1]:end[1]]))
  tv_Xa = np.hstack((v_Xa[:, start[0]:end[0]], v_Xa[:, start[1]:end[1]]))
  tRv_Xa = np.hstack((Rv_Xa[:, start[0]:end[0]], Rv_Xa[:, start[1]:end[1]]))
  tX_names = np.hstack((X_names[start[0]:end[0]], X_names[start[1]:end[1]]))
  for s in range(2, len(start)):
    tXa = np.hstack((tXa[:, :], Xa[:, start[s]:end[s]]))
    tRXa = np.hstack((tRXa[:, :], RXa[:, start[s]:end[s]]))
    tv_Xa = np.hstack((tv_Xa[:, :], v_Xa[:, start[s]:end[s]]))
    tRv_Xa = np.hstack((tRv_Xa[:, :], Rv_Xa[:, start[s]:end[s]]))
    tX_names = np.hstack((tX_names[:], X_names[start[s]:end[s]]))
  # tXa = Xa[:, start[0]:end[0]]
  # tRXa = RXa[:, start[0]:end[0]]
  # tv_Xa = v_Xa[:, start[0]:end[0]]
  # tRv_Xa = Rv_Xa[:, start[0]:end[0]]
  # tX_names = X_names[start[0]:end[0]]
  return tXa, tRXa, tv_Xa, tRv_Xa, tX_names

def analyze_metrics():
  EXPE = 5
  n_av      = metaseg.get("NUM_LASSO_AVERAGES")
  n_steps   = metaseg.get("NUM_LASSO_LAMBDAS")
  num_cores = min(n_av,metaseg.get("NUM_CORES"))

  metrics   = concatenate_metrics( save=False )
  nclasses  = np.max( metrics["class"] ) + 1
  v_metrics   = concatenate_metrics( train=False, save=False )
  v_nclasses  = np.max( v_metrics["class"] ) + 1
  print(nclasses, "UUUUUUUUUUUUUUUUUUUUUUUU")
  #print("^^^^^^", np.min(metrics["S"]))
  # print("&&&&&&&", np.max( v_metrics["class"] ))
  # print("******", np.array(v_metrics["class"]).shape)
  # print("$$$$$$", np.sum(v_metrics["class"]))
#   for i in metrics:
#     plt.clf()
#     plt.xlabel(f"{i}")
#     plt.ylabel("IOU")
#     #print(np.min(Xa[:, i]), "*********", X_names[i])
#     plt.scatter(metrics[i], metrics["iou"], 5)
#     # if i == "class":
#       # print(metrics[i][:1000])
#     plt.savefig(f'./expe{5}/co/{i}.png')
  # for i in v_metrics:
  #   plt.clf()
  #   plt.xlabel(f"{i}")
  #   plt.ylabel("IOU")
    # if i == "class":
    #   np.savetxt("./expe3/g.txt", v_metrics[i])
    #print(np.min(Xa[:, i]), "*********", X_names[i])
    # plt.scatter(v_metrics[i], v_metrics["iou"], 5)
    # plt.savefig(f'./expe{EXPE}/cov/{i}.png')
  
  # print(np.max(metrics['cprob0']), np.max(metrics['cprob1']))
  Xa, RXa, classes, Rclasses, ya, y0a, X_names, class_names = metrics_to_dataset( metrics, nclasses )
  v_Xa,Rv_Xa, v_classes, Rv_classes, v_ya, v_y0a, v_X_names, v_class_names = metrics_to_dataset( v_metrics, v_nclasses )
  
  Xa = np.concatenate( (Xa,classes), axis=-1 )
  RXa = np.concatenate( (RXa,Rclasses), axis=-1 )
  X_names += class_names
  v_Xa = np.concatenate( (v_Xa,v_classes), axis=-1 )
  Rv_Xa = np.concatenate( (Rv_Xa, Rv_classes), axis=-1 )
  v_X_names += v_class_names

  print(Xa.shape, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
  Xa, RXa, v_Xa, Rv_Xa, X_names = select_feature(Xa, RXa, v_Xa, Rv_Xa, X_names)
  # print(np.max(Xa[:,0]), np.max(Xa[:,1]), np.max(Xa[:,2]), np.max(Xa[:,3]), np.max(Xa[:,4]), np.max(Xa[:,5]))
  # for i in range(6):
  #   plt.clf()
  #   plt.xlabel(f"{X_names[i]}")
  #   plt.ylabel("IOU")
  #   # print(np.min(Xa[:, i]), "*********", X_names[i])
  #   plt.scatter(Xa[:, i], ya[:])
  #   plt.savefig(f'./expe{EXPE}/co/ori{i}.png')
  # Xa = np.exp(Xa)
  # print(np.max(Xa))
  # RXa = np.exp(RXa)
  # v_Xa = np.exp(v_Xa)
  # Rv_Xa = np.exp(Rv_Xa)

  for i in range(len(X_names)):
    plt.clf()
    plt.xlabel(f"{X_names[i]}")
    plt.ylabel("IOU")
    # print(np.min(Xa[:, i]), "*********", X_names[i])
    plt.scatter(Xa[:, i], ya[:])
    plt.savefig(f'./expe{EXPE}/co/{i}.png')
  
  # for i in range(17):
  #   plt.clf()
  #   plt.xlabel(f"{v_X_names[i]}")
  #   plt.ylabel("IOU")
  #   plt.scatter(v_Xa[:, i], v_ya[:])
  #   plt.savefig(f'./expe{EXPE}/cov/{i}.png')

  alphas           = get_alphas( n_steps, min_pow = -4.2, max_pow = 0.8 )
  stats            = init_stats( n_av, alphas, X_names )
  single_run_stats = init_stats( n_av, alphas, X_names )
  
  p = Pool(num_cores)
  
  p_args = [ ( Xa, RXa, ya, y0a, \
               v_Xa, Rv_Xa, v_ya, v_y0a, \
               alphas, X_names, single_run_stats, run ) for run in range(n_av) ]
  single_run_stats = p.starmap( fit_model_run, p_args )
  
  stats = merge_stats( stats, single_run_stats, n_av )
  
  mean_stats, _ = dump_stats( stats, metrics )
  
  generate_lasso_plots( stats, mean_stats, X_names, class_names )



if __name__ == '__main__':
  
  main()
      
    