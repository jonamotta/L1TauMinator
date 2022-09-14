from tensorflow.keras.initializers import RandomNormal as RN
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras import QConv2DBatchnorm, QActivation, QDense
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
from qkeras import qlayers
import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import sys
import os

np.random.seed(77)

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def inspectWeights(model, which):
    if which=='kernel': idx=0
    if which=='bias':   idx=1

    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[idx].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10,10))
    bins = np.linspace(-0.4, 0.4, 50)
    plt.hist(histosW,bins,histtype='step',stacked=True,label=labelsW)
    plt.legend(frameon=False,loc='upper left', fontsize=16)
    plt.ylabel('Recurrence')
    plt.xlabel('Weight value')
    plt.xlim(-0.7,0.5)
    plt.yscale('log')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQCNNIdentifier_plots/modelSparsity_'+which+'.pdf')
    plt.close()


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="5x9")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauQCNNIdentifier_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    Y = np.load(indir+'/Y_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']

    
    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauQCNNIdentifier_plots/training.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        x = images
        x = QConv2DBatchnorm(16, wndw, input_shape=(N, M, 3), kernel_initializer=RN(seed=7), use_bias=False,
                                                                     kernel_quantizer='quantized_bits(6,0,alpha=1)',  bias_quantizer='quantized_bits(6,0,alpha=1)',
                                                                     name='CNNpBNlayer1')(x)
        x = QActivation('quantized_relu(6,0)', name='reluCNNlayer1')(x)
        x = layers.MaxPooling2D(wndw, name='CNNlayer2')(x)
        x = QConv2DBatchnorm(24, wndw, kernel_initializer=RN(seed=7), use_bias=False,
                                              kernel_quantizer='quantized_bits(6,0,alpha=1)',  bias_quantizer='quantized_bits(6,0,alpha=1)',
                                              name='CNNpBNlayer3')(x)
        x = QActivation('quantized_relu(6,0)', name='reluCNNlayer3')(x)
        x = layers.Flatten(name="CNNflatened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions])
        x = QDense(32, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name='DNNlayer1')(x)
        x = QActivation('quantized_relu(6,0)', name='reluDNNlayer1')(x)
        x = QDense(16, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name='DNNlayer2')(x)
        x = QActivation('quantized_relu(6,0)', name='reluDNNlayer2')(x)
        x = QDense(1, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name="DNNout")(x)
        x = layers.Activation('sigmoid', name='sigmoidDNNout')(x)
        TauIdentified = x

        TauQIdentifierModel = keras.Model([images, positions], TauIdentified, name='TauCNNIdentifier')

        metrics2follow = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.AUC()]
        TauQIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                   metrics=metrics2follow,
                                   run_eagerly=True)

        # from qkeras.autoqkeras.utils import print_qmodel_summary
        # print_qmodel_summary(TauQCalibratorModel)        
        # exit()

        ############################## Model training ##############################

        history = TauQIdentifierModel.fit([X1, X2], Y, epochs=30, batch_size=1024, verbose=1, validation_split=0.2)

        TauQIdentifierModel.save(outdir + '/TauQCNNIdentifier')

        for metric in history.history.keys():
            if 'val_' in metric: continue

            plt.plot(history.history[metric], label='Training dataset', lw=2)
            plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            if metric=='loss': plt.legend(loc='upper right')
            else:              plt.legend(loc='lower right')
            mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
            plt.savefig(outdir+'/TauQCNNIdentifier_plots/'+metric+'.pdf')
            plt.close()

        ############################## Make split CNN and DNN models ##############################

        image_in = TauQIdentifierModel.get_layer(index=0).get_output_at(0)
        flat_out = TauQIdentifierModel.get_layer(name='middleMan').get_output_at(0)
        QCNNmodel = tf.keras.Model([image_in, positions], flat_out)
        QCNNmodel.save(outdir + '/QCNNmodel', include_optimizer=False)

        # flat_in = TauIdentifierModel.get_layer(name='middleMan').get_output_at(0)
        # id_out  = TauIdentifierModel.get_layer(name='sigmoidDNNout').get_output_at(0)
        # QDNNmodel = tf.keras.Model(flat_in.inputs, id_out)
        # DNNmodel.save(outdir + '/QDNNmodel', include_optimizer=False)

    else:
        TauQIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauQCNNIdentifier', compile=False)

    ############################## Model validation ##############################

    # load non-quantized model
    TauIdentifierModel = keras.models.load_model(indir+'/TauCNNIdentifier', compile=False)

    X1_valid = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load(indir+'/Y_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    train_Qident = TauQIdentifierModel.predict([X1, X2])
    QFPRtrain, QTPRtrain, QTHRtrain = metrics.roc_curve(Y, train_Qident)
    QAUCtrain = metrics.roc_auc_score(Y, train_Qident)

    valid_Qident = TauQIdentifierModel.predict([X1_valid, X2_valid])
    QFPRvalid, QTPRvalid, QTHRvalid = metrics.roc_curve(Y_valid, valid_Qident)
    QAUCvalid = metrics.roc_auc_score(Y_valid, valid_Qident)

    train_ident = TauIdentifierModel.predict([X1, X2])
    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Y, train_ident)
    AUCtrain = metrics.roc_auc_score(Y, train_ident)

    valid_ident = TauIdentifierModel.predict([X1_valid, X2_valid])
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Y_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Y_valid, valid_ident)

    inspectWeights(TauQIdentifierModel, 'kernel')
    # inspectWeights(TauQIdentifierModel, 'bias')

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.plot(QTPRtrain, QFPRtrain, label='Training ROC - Quantized, AUC = %.3f' % (QAUCtrain),   color='blue',lw=2, ls='--')
    plt.plot(QTPRvalid, QFPRvalid, label='Validation ROC - Quantized, AUC = %.3f' % (QAUCvalid), color='green',lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    # plt.xlim(0.8,1.01)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQCNNIdentifier_plots/validation_roc.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.plot(QTPRtrain, QFPRtrain, label='Training ROC - Quantized, AUC = %.3f' % (QAUCtrain),   color='blue',lw=2, ls='--')
    plt.plot(QTPRvalid, QFPRvalid, label='Validation ROC - Quantized, AUC = %.3f' % (QAUCvalid), color='green',lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.ylim(0.03,1)
    plt.yscale('log')
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQCNNIdentifier_plots/validation_roc_zoomed.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['Qscore'] = valid_Qident.ravel()
    df['true']  = Y_valid.ravel()
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), label='Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), label='PU', color='red', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==1]['Qscore'], bins=np.arange(0,1,0.05), label='Tau - Quantized Model', color='green', density=True, histtype='step', lw=2, ls='--')
    plt.hist(df[df['true']==0]['Qscore'], bins=np.arange(0,1,0.05), label='PU - Quantized Model', color='red', density=True, histtype='step', lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQCNNIdentifier_plots/CNN_score.pdf')
    plt.close()


    ############################## Feature importance ##############################

    # since we have two inputs we pass a list of inputs to the explainer
#    explainer = shap.GradientExplainer(TauQIdentifierModel, [X1, X2])
#
#    # we explain the model's predictions on the first three samples of the test set
#    shap_values = explainer.shap_values([X1_valid[:3], X2_valid[:3]])
#
#    # since the model has 10 outputs we get a list of 10 explanations (one for each output)
#    print(len(shap_values))
#
#    # since the model has 2 inputs we get a list of 2 explanations (one for each input) for each output
#    print(len(shap_values[0]))
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the first input (this is the feed forward input)
#    shap.image_plot([shap_values[i][0] for i in range(len(shap_values))], X1_valid[:3], show=False)
#    plt.savefig(outdir+'/TauQCNNIdentifier_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauQCNNIdentifier_plots/shap1.pdf')
#    plt.close()

# restore normal output
sys.stdout = sys.__stdout__
