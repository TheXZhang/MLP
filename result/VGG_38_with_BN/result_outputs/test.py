import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df38_bn2 = pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_BN/result_outputs/summary.csv')
df38_bn2['epoch']= list(range(1, 101))
df08 = pd.read_csv('C:/Users/zhang/Desktop/mlpractical/VGG_08/result_outputs/summary.csv')
df08['epoch']= list(range(1, 102))
df38_res = pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_ResNet/VGG_38_experiment/result_outputs/summary.csv')
df38_res['epoch']= list(range(1, 101))


baseline=pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_ResNetBN/VGG_38_experiment/result_outputs/summary.csv')
baseline['epoch']= list(range(1, 101))

tem=pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_ResNetBN_random_erasing/VGG_38_experiment/result_outputs/summary.csv')
tem['epoch']= list(range(1, 101))

tem2=pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_ResNetBN_rotate/VGG_38_experiment/result_outputs/summary.csv')
tem2['epoch']= list(range(1, 101))


decay=pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_ResNetBN_0.001/VGG_38_experiment/result_outputs/summary.csv')
decay['epoch']= list(range(1, 101))



dropout=pd.read_csv('C:/Users/zhang/Desktop/mlpractical/result/VGG_38_with_ResNetBN_dropOut/VGG_38_experiment/result_outputs/summary.csv')
dropout['epoch']= list(range(1, 101))
# print(ep)
# print(df38_bn2)
# print(df08)
# print(df38_res)


# plt.plot('epoch', 'val_acc',data=baseline, label='baseline_valid_acc')
# plt.plot('epoch', 'val_acc',data=tem, label='random_erasing_valid_acc')
# plt.plot('epoch', 'val_acc',data=tem2, label='random_rotate_valid_acc')


# plt.plot('epoch', 'train_acc',data=baseline, label='baseline_train_acc')
# plt.plot('epoch', 'val_acc',data=baseline, label='baseline_val_acc')
plt.plot('epoch', 'train_acc',data=tem, label='random_erasing_train_acc')
plt.plot('epoch', 'val_acc',data=tem, label='random_erasing_val_acc')
plt.plot('epoch', 'train_acc',data=dropout, label='dropOut_train_acc')
plt.plot('epoch', 'val_acc',data=dropout, label='dropOut_val_acc')

plt.xlabel('epoch number', fontsize=12)

plt.ylabel('accuracy', fontsize=12)

# plt.plot('epoch', 'train_acc',data=df38_bn2, label='VGG_38_BN_train_acc')
# plt.plot('epoch', 'train_acc',data=df38_res, label='VGG_38_ResNet_train_acc')
# plt.plot('epoch', 'train_acc',data=tem, label='VGG_38_combined_train_acc')
# plt.plot('epoch', 'train_acc',data=df08,label='VGG_08_train_acc')

# plt.plot('epoch', 'val_acc',data=df38_bn2, label='VGG_38_BN_val_acc')
# plt.plot('epoch', 'val_acc',data=df38_res, label='VGG_38_ResNet_val_acc')
# plt.plot('epoch', 'val_acc',data=tem, label='VGG_38_combined_val_acc')
# plt.plot('epoch', 'val_acc',data=df08,label='VGG_08_val_acc')



plt.legend()
plt.show()
# -------------------------------38 batch normalization
#plt.plot('epoch', 'train_acc',data=df38_bn2, label='VGG_38 BN train_acc')
#plt.plot('epoch', 'val_acc',data=df38_bn2, label='VGG_38 BN val_acc')
#plt.plot('epoch', 'train_acc',data=df08,label='VGG_08 train_acc')
#plt.plot('epoch', 'val_acc',data=df08, label='VGG_08 val_acc')
#plt.legend()
# plt.savefig('acc.png')
#plt.show()

# plt.plot('epoch', 'train_loss',data=df38_bn2, label='VGG_38 BN train_loss')
# plt.plot('epoch', 'val_loss',data=df38_bn2, label='VGG_38 BN val_loss')
# plt.plot('epoch', 'train_loss',data=df08,label='VGG_08 train_loss')
# plt.plot('epoch', 'val_loss',data=df08, label='VGG_08 val_loss')
# plt.legend()
# # plt.savefig('loss.png')
# plt.show()
#
# # -------------------------------VGG_38_res
# plt.plot('epoch', 'train_acc',data=df38_res, label='VGG_38 res train_acc')
# plt.plot('epoch', 'val_acc',data=df38_res, label='VGG_38 res val_acc')
# plt.plot('epoch', 'train_acc',data=df08,label='VGG_08 train_acc')
# plt.plot('epoch', 'val_acc',data=df08, label='VGG_08 val_acc')
# plt.legend()
# # plt.savefig('acc.png')
# plt.show()
# plt.plot('epoch', 'train_loss',data=df38_res, label='VGG_38 res train_loss')
# plt.plot('epoch', 'val_loss',data=df38_res, label='VGG_38 res val_loss')
# plt.plot('epoch', 'train_loss',data=df08,label='VGG_08 train_loss')
# plt.plot('epoch', 'val_loss',data=df08, label='VGG_08 val_loss')
# plt.legend()
# # plt.savefig('loss.png')
# plt.show()
