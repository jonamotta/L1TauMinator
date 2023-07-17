from hls4ml.templates.vivado_accelerator.alveo.python_drivers.axi_stream_driver import NeuralNetworkOverlay
import numpy as np
import os

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)


directory = './TauMinator/CE'
cnnout = np.load(directory+'/tensors_AlveoTest/CONV_output.npz')['arr_0']
X2 = np.load(directory+'/tensors_AlveoTest/posits_train.npz')['arr_0']
X3 = np.load(directory+'/tensors_AlveoTest/shapes_train.npz')['arr_0']

scaler = load_obj(indir+'/CL3D_features_scaler/cl3d_features_scaler.pkl')
X3 = scaler.transform(X3)

# features2use = pt, eta, showerlength, coreshowerlength, spptot, szz, srrtot, meanz
features2use = [0, 2, 4, 5, 9, 11, 12, 16]
X3 = X3[:,features2use]

middleMan = np.concatenate([cnnout, X2, X3], axis=1)

DNN_input_shape = (len(cnnout), 24+2+len(features2use))
DNN_output_shape = (len(cnnout), 1)

ID_DNN_bin = NeuralNetworkOverlay(directory+'/ID_DNNmodel_HLS_XCU200deployment/xclbin_files/myproject_kernel.xclbin')
ID_DNN_bin.allocate_mem(DNN_input_shape, DNN_output_shape)
idout = ID_DNN_bin.predict(middleMan, DNN_output_shape)
ID_DNN_bin.input_buffer.freebuffer()
ID_DNN_bin.output_buffer.freebuffer()
ID_DNN_bin.free_overlay()

np.savez_compressed(directory+'/tensors_AlveoTest/ID_output.npz', idout)