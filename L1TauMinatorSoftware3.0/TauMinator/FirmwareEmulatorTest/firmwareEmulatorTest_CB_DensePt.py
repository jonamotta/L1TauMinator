from hls4ml.templates.vivado_accelerator.alveo.python_drivers.axi_stream_driver import NeuralNetworkOverlay
import numpy as np
import os


directory = './TauMinator/CB'
cnnout = np.load(directory+'/tensors_AlveoTest/CONV_output.npz')['arr_0']
X2 = np.load(directory+'/tensors_AlveoTest/posits_train.npz')['arr_0']

middleMan = np.concatenate([cnnout, X2], axis=1)

DNN_input_shape = (len(cnnout), 24+2)
DNN_output_shape = (len(cnnout), 1)

CAL_DNN_bin = NeuralNetworkOverlay(directory+'/CAL_DNNmodel_HLS_XCU200deployment/xclbin_files/myproject_kernel.xclbin')
CAL_DNN_bin.allocate_mem(DNN_input_shape, DNN_output_shape)
ptout = CAL_DNN_bin.predict(middleMan, DNN_output_shape)
CAL_DNN_bin.input_buffer.freebuffer()
CAL_DNN_bin.output_buffer.freebuffer()
CAL_DNN_bin.free_overlay()

np.savez_compressed(directory+'/tensors_AlveoTest/PT_output.npz', ptout)