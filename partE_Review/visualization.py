import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math




class Calculate_accuracy(nn.Module):
    def __init__(self, error):
        super(Calculate_accuracy, self).__init__()
        self.error = error
        # units: meters

    def forward(self, preds, trues):
        s = preds[:,:,0] /180 * np.pi
        lon1, lat1, lon2, lat2 = map(lambda x: x / 180 * np.pi, [preds[:,:,0], preds[:,:,1], trues[:,:,0], trues[:,:,1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(a ** 0.5)
        r = 6371000  # meter
        dis = c * r

        right = np.sum(dis<self.error)
        all = np.size(dis)
        acc = right/all
        return acc


# args
map_root = '/home/user/Documents/Yangkaisen/OE-GMLTP/GCN_Informer_test/map/map_Aarea.png'
Preds = np.load('./output_f4_16/GAN/Preds.npy')
Reals = np.load('./output_f4_16/GAN/Reals.npy')

K = 20
observed = 16

# show
mean_true = Reals.mean(axis=(0, 1), keepdims = True)[:, :, :2]
std_true = Reals.std(axis=(0, 1), keepdims = True)[:, :, :2]
if Preds.ndim ==4:
    mean_true = mean_true[np.newaxis,:, :, :2]
    std_true = std_true[np.newaxis, :, :, :2]
    Preds = Preds[:,:,:,:2] * std_true + mean_true
else:
    Preds = Preds[:, :, :2] * std_true + mean_true

Reals = Reals[:,:,:2]

keep_ship = [7,9,11,14,16, 17,25,21,28,29,41,43,44,46,50,56,57] #+ [k for k in range(40,50)]

plt.scatter(Reals[:, keep_ship, 0], Reals[:, keep_ship, 1],c='b',s=3)
if Preds.ndim ==4:
    plt.scatter(Preds[:, keep_ship, :,0], Preds[:, keep_ship, :,1],c='r',alpha=0.02, s=20)
else:
    plt.scatter(Preds[:, keep_ship,  0], Preds[:, keep_ship,  1], c='r', alpha=0.8, s=3)



imp = plt.imread(map_root)
plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
plt.show()
print(';')



#cal_acc = Calculate_accuracy(error=120)
#acc = cal_acc(Preds, Reals)

# hongkong, 16-16pred &ade0.052: err&acc: [20m-0.097, 50m-0.40, 100m-0.72, 120m-0.80]
#           16-16pred &ade0.036: err&acc: [20m-0.24, 50m-0.64, 100m-0.86, 120m-0.90]
# shanghai  16-16pred &ade0.023: err&acc: [20m-0.40, 50m-0.78, 100m-0.93, 120m-0.95]
#           16-16pred &ade0.018: err&acc: [20m-0.65, 50m-0.91, 100m-0.98, 120m-0.99]
