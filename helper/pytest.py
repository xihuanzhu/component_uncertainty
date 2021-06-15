#
import numpy as np
import pickle
import matplotlib.pyplot as plt
# info = pickle.load(open( "/home/zhuxihuan/code/MetaSeg/expe3/train_measure/stats/best-loss-037epoch.bin/stats.p", "rb" ))
# print(info)
# if True:
#     print("fjdf")
# if False:
#     print("*****")
# from sklearn.linear_model import Ridge
# reg = Ridge(alpha=.5)
# reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
# print(reg.coef_)
# print(reg.intercept_)
# for i in range(20):
#     components = pickle.load( open( f"/home/zhuxihuan/code/MetaSeg/expe1/train_measure/components/best-loss-037epoch.bin/components{i}.p", "rb" ) )
#     tmetrics   = pickle.load( open(f"/home/zhuxihuan/code/MetaSeg/expe2/train_measure/metrics/best-loss-037epoch.bin/metrics{i}.p", "rb" ) )
#     print("*******npmax: ", np.max(components), "npmin: ", np.min(components))
#    # print("npmax: ", np.max(tmetrics), "npmin: ", np.min(components))

# print(np.max([[2,3], [1,  9]]))
# cmap=plt.get_cmap('tab20')
# print(cmap(0))
t = []
A = [1,2,3,3,3,3,3,3]
t= np.hstack([t[:], A[:]])
print(t)