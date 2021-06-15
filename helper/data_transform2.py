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
   train_f = 0
   print(len(os.listdir(pred_dir))/3)
   for sdir in os.listdir(pred_dir):
      if sdir.endswith("image.npy") or sdir.endswith("label.npy"):
         continue
      train_f = train_f + 1
      if train_f > 55:
         continue
      #print(sdir)
      sdir = sdir[:-9]
      print(sdir)
      # if sdir.startswith('.'):
      #    continue
      pred = np.load(os.path.join(pred_dir, sdir) + "_pred.npy")
      gt = np.load(gt_dir + sdir + "_label.npy")
      if not os.path.exists(save_dir + "GT/"):
         os.makedirs(save_dir + "GT/")
      if not os.path.exists(save_dir + "probs/"):
         os.makedirs(save_dir + "probs/")
      if not os.path.exists(save_dir + "IMG/"):
         os.makedirs(save_dir + "IMG/")

      print(pred.shape, gt.shape)
      for i in range(pred.shape[0]):
         one_gt = np.squeeze(gt[i])
         #print(one_gt.shape)
         one_img = np.load(img_dir + sdir + "/image_" +str(i).rjust(3, '0') +".npy")
         #print(one_img.shape)
         one_img = np.squeeze(one_img)
         opred = np.squeeze(pred[i])
         tpred = 1 - opred
         one_pred = np.array([tpred, opred]).transpose(1, 2, 0)
         #print(i, "****",  sum + i)
         np.save(f"{save_dir}/GT/{sum + i}.npy", one_gt)
         np.save(f"{save_dir}/probs/{sum + i}.npy", one_pred) 
         np.save(f"{save_dir}/IMG/{sum + i}.npy", one_img)
   
      sum = sum + pred.shape[0]
      f = open(f"{save_dir}out.txt", "a+")    # 打开文件以便写入
      print("patience's name is ", sdir, " num is ", pred.shape[0], file=f)
      f.close  #  关闭文件


gt_dir = "/raid/zhuxihuan/data/body/wjc_revised_data/"
img_dir = "/raid/wjc/data/guangxi/mr_numpy/"
pred_dir = "/raid/zhuxihuan/data/body/wjc_revised_data/"
save_dir = "/raid/zhuxihuan/data/body/expe6/train_data/"
if not os.path.exists(save_dir):
   os.makedirs(save_dir)
f = open(f"{save_dir}out.txt", "w+") 
f.close 
data_transform(gt_dir, img_dir, pred_dir, save_dir)


