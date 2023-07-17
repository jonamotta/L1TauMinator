from hls4ml.templates.vivado_accelerator.alveo.python_drivers.axi_stream_driver import NeuralNetworkOverlay
import numpy as np
import os


directory = './TauMinator/CB'
X1 = np.load(directory+'/tensors_AlveoTest/images_train.npz')['arr_0']

CNN_input_shape = (len(X1), 5, 9, 2)
CNN_output_shape = (len(X1), 24)

CNN_bin = NeuralNetworkOverlay(directory+'/CNNmodel_HLS_XCU200deployment/xclbin_files/myproject_kernel.xclbin')
CNN_bin.allocate_mem(CNN_input_shape, CNN_output_shape)
cnnout = CNN_bin.predict(X1, CNN_output_shape)
CNN_bin.input_buffer.freebuffer()
CNN_bin.output_buffer.freebuffer()
CNN_bin.free_overlay()

np.savez_compressed(directory+'/tensors_AlveoTest/CONV_output.npz', cnnout)