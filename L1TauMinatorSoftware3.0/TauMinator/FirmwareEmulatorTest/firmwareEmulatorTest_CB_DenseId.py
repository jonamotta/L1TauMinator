from hls4ml.templates.vivado_accelerator.alveo.python_drivers.axi_stream_driver import NeuralNetworkOverlay
import numpy as np
import os


directory = './TauMinator/CB'
cnnout = np.load(directory+'/tensors_AlveoTest/CONV_output.npz')['arr_0']
X2 = np.load(directory+'/tensors_AlveoTest/posits_train.npz')['arr_0']

middleMan = np.concatenate([cnnout, X2], axis=1)

DNN_input_shape = (len(middleMan), 24+2)
DNN_output_shape = (len(middleMan), 1)

ID_DNN_bin = NeuralNetworkOverlay(directory+'/ID_DNNmodel_HLS_XCU200deployment/xclbin_files/myproject_kernel.xclbin')
ID_DNN_bin.allocate_mem(DNN_input_shape, DNN_output_shape)
idout = ID_DNN_bin.predict(middleMan, DNN_output_shape)
ID_DNN_bin.input_buffer.freebuffer()
ID_DNN_bin.output_buffer.freebuffer()
ID_DNN_bin.free_overlay()

np.savez_compressed(directory+'/tensors_AlveoTest/ID_output.npz', idout)