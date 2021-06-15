import numpy as np
import os
# probs = np.ones((512,512,2))
# for i in range(probs.shape[0]):
#     if i <= 5:
#        probs[i,:,0] = (i+1)/probs.shape[0]
#     else:
#        probs[i,:,0] = (10-i+1)/probs.shape[0]
#     probs[i,:,1] = 1 - probs[i,:,0]

# gt = np.random.random((512, 512))
# img = np.random.random((512, 512))
# np.save("/home/zhuxihuan/code/MetaSeg/data/GT/0.npy", gt)
# np.save("/home/zhuxihuan/code/MetaSeg/data/probs/0.npy", probs)
# np.save("/home/zhuxihuan/code/MetaSeg/data/IMG/0.npy", img)
# gt = np.load("/raid/zhuxihuan/data/body/test_npy_data/BodyV2_9-BodyV2_9/mask.npy")
# img = np.load("/raid/zhuxihuan/data/body/test_npy_data/BodyV2_9-BodyV2_9/slice.npy")
# pred = np.load("/home/zhuxihuan/code/method/two_pre_test_dir/BodyV2_9-BodyV2_9/pmask.npy")
# print(gt.shape)
# print(img.shape)
# print(pred.shape)

# gt_path = "/home/zhuxihuan/code/MetaSeg/al_data/GT"
# probs_path = "/home/zhuxihuan/code/MetaSeg/al_data/probs"
# img_path = "/home/zhuxihuan/code/MetaSeg/al_data/IMG"
def data_transform(gt_dir, img_dir, pred_dir, save_dir):
   sum = 0
   for sdir in os.listdir(pred_dir):
      print(sdir)
      if sdir.startswith('.'):
         continue
      gt = np.load(os.path.join(gt_dir, sdir) + "/mask.npy")
      #print(gt.shape[0])
      img = np.load(os.path.join(img_dir, sdir) + "/slice.npy")
      pred = np.load(os.path.join(pred_dir, sdir) + "/mean_mask.npy")

      if not os.path.exists(save_dir + "GT/"):
         os.makedirs(save_dir + "GT/")
      if not os.path.exists(save_dir + "probs/"):
         os.makedirs(save_dir + "probs/")
      if not os.path.exists(save_dir + "IMG/"):
         os.makedirs(save_dir + "IMG/")

      for i in range(gt.shape[0]):
         one_gt = np.squeeze(gt[i])
         one_img = np.squeeze(img[i])
         opred = np.squeeze(pred[i])
         tpred = 1 - opred
         one_pred = np.array([tpred, opred]).transpose(1, 2, 0)
         #print(i, "****",  sum + i)
         np.save(f"{save_dir}/GT/{sum + i}.npy", one_gt)
         np.save(f"{save_dir}/probs/{sum + i}.npy", one_pred) 
         np.save(f"{save_dir}/IMG/{sum + i}.npy", one_img)
   
      sum = sum + gt.shape[0]
      f = open(f"{save_dir}out.txt", "a+")    # 打开文件以便写入
      print("patience's name is ", sdir, " num is ", gt.shape[0], file=f)
      f.close  #  关闭文件


gt_dir = "/raid/zhuxihuan/data/body/test_npy_data/"
img_dir = "/raid/zhuxihuan/data/body/test_npy_data/"
pred_dir = "/raid/zhuxihuan/data/body/mc_dropout_BCE_pre_test5_dir/"
save_dir = "/raid/zhuxihuan/data/body/expe5/val_data/"
if not os.path.exists(save_dir):
   os.makedirs(save_dir)
f = open(f"{save_dir}out.txt", "w+") 
f.close 
data_transform(gt_dir, img_dir, pred_dir, save_dir)


