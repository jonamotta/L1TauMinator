from tensorflow.keras.initializers import RandomNormal as RN
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras import QConv2DBatchnorm, QActivation, QDense
from sklearn.preprocessing import StandardScaler
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
from qkeras import qlayers
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

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
    plt.savefig(outdir+'/TauQDNNIdentifier_plots/modelSparsity'+which+'.pdf')
    plt.close()


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTIdentifierTraining'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNIdentifierTraining'+options.inTag
    os.system('mkdir -p '+outdir+'/TauQDNNIdentifier_plots')

    dfTr = pd.read_pickle(indir+'/X_Ident_BDT_forIdentifier.pkl')
    dfTr['cl3d_abseta'] = abs(dfTr['cl3d_eta']).copy(deep=True)

    feats = ['cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_localAbsMeanZ']

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(dfTr[feats]), columns=feats)

    TrTensorizedInput  = scaled[feats].to_numpy()
    TrTensorizedTarget = dfTr['targetId'].to_numpy()

    ############################# Models definition ##############################

    # This model identifies the tau object:
    #    - one DNN that takes the features of the CL3D clusters

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauQDNNIdentifier_plots/training.log')
        print(options)

        features = keras.Input(shape=len(feats), name='CL3DFeatures')

        x = features
        x = QDense(16, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name='DNNlayer')(x)
        x = QActivation('quantized_relu(16,6)', name='RELU_DNNlayer')(x)
        x = QDense(1, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name="DNNout")(x)
        x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
        TauIdentified = x

        TauQIdentifierModel = keras.Model(features, TauIdentified, name='TauQDNNIdentifier')

        metrics2follow = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        TauQIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(),
                                   metrics=metrics2follow,
                                   run_eagerly=True)

        # print(TauQIdentifierModel.summary())
        # exit()

        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauQIdentifierModel.fit(TrTensorizedInput, TrTensorizedTarget, epochs=200, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

        TauQIdentifierModel.save(outdir + '/TauQDNNIdentifier')

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauQDNNIdentifier_plots/'+metric+'.pdf')
                plt.close()

            else:
                if 'val_' in metric: continue

                plt.plot(history.history[metric], label='Training dataset', lw=2)
                plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
                plt.xlabel('Epoch')
                if metric=='loss':
                    plt.ylabel('Loss')
                    plt.legend(loc='upper right')
                elif metric=='auc':
                    plt.ylabel('AUC')
                    plt.legend(loc='lower right')
                elif metric=='binary_accuracy':
                    plt.ylabel('Binary accuracy')
                    plt.legend(loc='lower right')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauQDNNIdentifier_plots/'+metric+'.pdf')
                plt.close()

    else:
        TauQIdentifierModel = keras.models.load_model(outdir+'/TauQDNNIdentifier', compile=False)


    ############################## Model validation ##############################

    # load non-quantized model
    TauIdentifierModel = keras.models.load_model(outdir+'/TauDNNIdentifier', compile=False)

    dfVal = pd.read_pickle(indir+'/X_Ident_BDT_forEvaluator.pkl')
    dfVal['cl3d_abseta'] = abs(dfVal['cl3d_eta']).copy(deep=True)

    scaled = pd.DataFrame(scaler.transform(dfVal[feats]), columns=feats)
    ValTensorizedInput  = scaled[feats].to_numpy()
    ValTensorizedTarget = dfVal['targetId'].to_numpy()

    train_ident = TauIdentifierModel.predict(TrTensorizedInput)
    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(TrTensorizedTarget, train_ident)
    AUCtrain = metrics.roc_auc_score(TrTensorizedTarget, train_ident)

    valid_ident = TauIdentifierModel.predict(ValTensorizedInput)
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(ValTensorizedTarget, valid_ident)
    AUCvalid = metrics.roc_auc_score(ValTensorizedTarget, valid_ident)

    train_Qident = TauQIdentifierModel.predict(TrTensorizedInput)
    QFPRtrain, QTPRtrain, QTHRtrain = metrics.roc_curve(TrTensorizedTarget, train_Qident)
    QAUCtrain = metrics.roc_auc_score(TrTensorizedTarget, train_Qident)

    valid_Qident = TauQIdentifierModel.predict(ValTensorizedInput)
    QFPRvalid, QTPRvalid, QTHRvalid = metrics.roc_curve(ValTensorizedTarget, valid_Qident)
    QAUCvalid = metrics.roc_auc_score(ValTensorizedTarget, valid_Qident)

    inspectWeights(TauQIdentifierModel, 'kernel')

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
    plt.savefig(outdir+'/TauQDNNIdentifier_plots/validation_roc.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.plot(QTPRtrain, QFPRtrain, label='Training ROC - Quantized, AUC = %.3f' % (QAUCtrain),   color='blue',lw=2, ls='--')
    plt.plot(QTPRvalid, QFPRvalid, label='Validation ROC - Quantized, AUC = %.3f' % (QAUCvalid), color='green',lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.ylim(0.01,1)
    plt.yscale('log')
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNIdentifier_plots/validation_roc_zoomed.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['Qscore'] = valid_Qident.ravel()
    df['true']  = ValTensorizedTarget.ravel()
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), label='Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), label='PU', color='red', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==1]['Qscore'], bins=np.arange(0,1,0.05), label='Tau - Quantized', color='green', density=True, histtype='step', lw=2, ls='--')
    plt.hist(df[df['true']==0]['Qscore'], bins=np.arange(0,1,0.05), label='PU - Quantized', color='red', density=True, histtype='step', lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'DNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNIdentifier_plots/DNN_score.pdf')
    plt.close()

# restore normal output
sys.stdout = sys.__stdout__

