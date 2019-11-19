import numpy as np
import scipy.signal
import time

inputs = np.arange(96).reshape((2, 3, 4, 4))
kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
conv_shape=((self.input_height-self.kernel_height+1),(self.input_width-self.kernel_width+1))
temp_matrix=np.zeros(conv_shape)
result_shape=(inputs.shape[0],self.num_output_channels,conv_shape)
result=np.zeros(result_shape)
for i in range(inputs.shape[0]):
    for j in range(inputs.shape[0]):
        for k in range(inputs.shape[1]):
            temp_matrix += scipy.signal.convolve2d(inputs[i][k],self.kernels[j][k],mode="valid")
        temp_matrix+=self.biases[j]
        result[i][j]=temp_matrix
        temp_matrix=np.zeros((3,3))





#
# true_grads_wrt_inputs = np.array(
#       [[[[ 147.,  319.,  305.,  162.],
#          [ 338.,  716.,  680.,  354.],
#          [ 290.,  608.,  572.,  294.],
#          [ 149.,  307.,  285.,  144.]],
#         [[  23.,   79.,   81.,   54.],
#          [ 114.,  284.,  280.,  162.],
#          [ 114.,  272.,  268.,  150.],
#          [  73.,  163.,  157.,   84.]],
#         [[-101., -161., -143.,  -54.],
#          [-110., -148., -120.,  -30.],
#          [ -62.,  -64.,  -36.,    6.],
#          [  -3.,   19.,   29.,   24.]]],
#        [[[  39.,   67.,   53.,   18.],
#          [  50.,   68.,   32.,   -6.],
#          [   2.,  -40.,  -76.,  -66.],
#          [ -31.,  -89., -111.,  -72.]],
#         [[  59.,  115.,  117.,   54.],
#          [ 114.,  212.,  208.,   90.],
#          [ 114.,  200.,  196.,   78.],
#          [  37.,   55.,   49.,   12.]],
#         [[  79.,  163.,  181.,   90.],
#          [ 178.,  356.,  384.,  186.],
#          [ 226.,  440.,  468.,  222.],
#          [ 105.,  199.,  209.,   96.]]]])
#
# kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
#
# grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
#
# #print(grads_wrt_outputs)
#
#
#
# padded_grads = np.pad(grads_wrt_outputs, [(0, 0), (0, 0), (1, 1), (1,1)], mode='constant')
#
# print (padded_grads)
# sub_shape = (2, 2)
# shape = padded_grads.shape[0:2] + tuple(np.subtract(padded_grads.shape[-2:], sub_shape) + 1) + sub_shape
# #print(shape)
# M = np.lib.stride_tricks.as_strided(padded_grads, shape=shape, strides=padded_grads.strides + padded_grads.strides[-2:])
# #print(M)
# dinputs = np.einsum('fcpq, bfyzpq -> bcyz', kernels, M)
# #print(dinputs)
