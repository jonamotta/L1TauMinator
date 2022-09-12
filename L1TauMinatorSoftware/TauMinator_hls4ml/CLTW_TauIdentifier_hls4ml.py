from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras.utils import _add_supported_quantized_objects
from optparse import OptionParser
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
# import plotting
import hls4ml
import shap
import sys
import os

np.random.seed(77)

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="5x9")
    parser.add_option('--sparsity',     dest='sparsity',  type=float,          default=0.5)
    (options, args) = parser.parse_args()
    print(options)

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    sparsityTag = str(options.sparsity).split('.')[0]+'p'+str(options.sparsity).split('.')[1]

    ############################## Get trained models ##############################

    # load non-pruned models
    TauIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier', compile=False)
    TauQIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauQCNNIdentifier', compile=False)

    # load pruned models
    # TauIdentifierModelPruned_ = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier'+sparsityTag+'Pruned', compile=False)
    # TauQIdentifierModelPruned_ = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauQCNNIdentifier'+sparsityTag+'Pruned', compile=False)

    # # remove pruning part of the model that has nothing to do with prediction
    # co = {}
    # _add_supported_quantized_objects(co)
    # co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    # TauIdentifierModelPruned  = strip_pruning(TauIdentifierModelPruned_)
    # TauQIdentifierModelPruned  = strip_pruning(TauQIdentifierModelPruned_)

    ############################## Pass non-quantized model through hls4ml ##############################

    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    # baseline model
    hls_config = hls4ml.utils.config_from_keras_model(TauIdentifierModel, granularity='name')
    hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['Model']['ReuseFactor'] = 1
    for Layer in hls_config['LayerName'].keys():
        hls_config['LayerName'][Layer]['Strategy'] = 'Latency'
        hls_config['LayerName'][Layer]['ReuseFactor'] = 1
    hls_config['LayerName']['sigmoidDNNout']['Strategy'] = 'Stable'
    # plotting.print_dict(hls_config)
    print(hls_config)

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = TauIdentifierModel
    cfg['OutputDir']  = 'TauIdentifierModel/'
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'
    
    hls_TauIdentifierModel = hls4ml.converters.keras_to_hls(cfg)
    hls_TauIdentifierModel.compile()

    hls4ml.utils.plot_model(hls_TauIdentifierModel, show_shapes=True, show_precision=True, to_file=None)
    hls4ml.model.profiling.numerical(model=TauIdentifierModel, hls_model=hls_TauIdentifierModel)

    # ############################## Pass quantized model through hls4ml ##############################

    # hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    # Qhls_config = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
    # Qhls_config['Model']['ReuseFactor'] = 1
    # Qhls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    # Qhls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
    # plotting.print_dict(Qhls_config)
      
    # cfg_q = hls4ml.converters.create_config(backend='Vivado')
    # cfg_q['IOType']     = 'io_stream' # Must set this if using CNNs!
    # cfg_q['HLSConfig']  = Qhls_config
    # cfg_q['KerasModel'] = TauQIdentifierModel
    # cfg_q['OutputDir']  = 'TauQIdentifierModel/'
    # cfg_q['XilinxPart'] = 'xcu250-figd2104-2L-e'
      
    # hls_TauQIdentifierModel = hls4ml.converters.keras_to_hls(cfg_q)
    # hls_TauQIdentifierModel.compile()

    # hls4ml.utils.plot_model(hls_TauQIdentifierModel, show_shapes=True, show_precision=True, to_file=None)
    # hls4ml.model.profiling.numerical(model=qmodel, hls_model=hls_TauQIdentifierModel)














