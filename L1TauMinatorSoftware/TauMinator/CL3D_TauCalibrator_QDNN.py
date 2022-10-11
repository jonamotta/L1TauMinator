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
import pickle
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
    bins = np.linspace(-5, 5, 100)
    plt.hist(histosW,bins,histtype='step',stacked=True,label=labelsW)
    plt.legend(frameon=False,loc='upper left', fontsize=16)
    plt.ylabel('Recurrence')
    plt.xlabel('Weight value')
    plt.xlim(-5,5)
    plt.yscale('log')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/modelSparsity'+which+'.pdf')
    plt.close()

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)


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

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTCalibratorTraining'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNCalibratorTraining'+options.inTag
    os.system('mkdir -p '+outdir+'/TauQDNNCalibrator_plots')

    dfTr = pd.read_pickle(indir+'/X_Calib_BDT_forCalibrator.pkl')

    feats = ['cl3d_pt', 'cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ']

    scaler = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNOptimization/dnn_features_scaler.pkl')
    scaled = pd.DataFrame(scaler.transform(dfTr[feats]), columns=feats)

    TrTensorizedInput  = scaled[feats].to_numpy()
    TrTensorizedTarget = dfTr['tau_visPt'].to_numpy()


    ############################# Models definition ##############################

    # This model calibrates the tau object:
    #    - one DNN that takes the features of the CL3D clusters

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauQDNNCalibrator_plots/training.log')
        print(options)

        features = keras.Input(shape=len(feats), name='CL3DFeatures')

        x = features
        x = QDense(16, use_bias=False, kernel_quantizer='quantized_bits(10,3,alpha=1)', name='DNNlayer')(x)
        x = QActivation('quantized_relu(16,6)', name='RELU_DNNlayer')(x)
        x = QDense(1, use_bias=False, kernel_quantizer='quantized_bits(10,3,alpha=1)', name="DNNout")(x)
        TauCalibrated = x

        TauQCalibratorModel = keras.Model(features, TauCalibrated, name='TauQDNNCalibrator')

        TauQCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   metrics=['RootMeanSquaredError'],
                                   run_eagerly=True)

        # print(TauQCalibratorModel.summary())
        # exit()

        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauQCalibratorModel.fit(TrTensorizedInput, TrTensorizedTarget, epochs=200, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

        TauQCalibratorModel.save(outdir + '/TauQDNNCalibrator')

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauQDNNCalibrator_plots/'+metric+'.pdf')
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
                plt.savefig(outdir+'/TauQDNNCalibrator_plots/'+metric+'.pdf')
                plt.close()

    else:
        TauQCalibratorModel = keras.models.load_model(outdir+'/TauQDNNCalibrator', compile=False)


    ############################## Model validation ##############################

    # load non-quantized model
    TauCalibratorModel = keras.models.load_model(outdir+'/TauDNNCalibrator', compile=False)

    dfVal = pd.read_pickle(indir+'/X_Calib_BDT_forEvaluator.pkl')

    scaled = pd.DataFrame(scaler.transform(dfVal[feats]), columns=feats)
    ValTensorizedInput  = scaled[feats].to_numpy()
    ValTensorizedTarget = dfVal['tau_visPt'].to_numpy()

    train_calib = TauCalibratorModel.predict(TrTensorizedInput)
    valid_calib = TauCalibratorModel.predict(ValTensorizedInput)

    train_Qcalib = TauQCalibratorModel.predict(TrTensorizedInput)
    valid_Qcalib = TauQCalibratorModel.predict(ValTensorizedInput)

    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = dfTr['cl3d_pt']
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['Qcalib_pt']  = train_Qcalib.ravel()
    dfTrain['gen_pt']     = dfTr['tau_visPt']
    dfTrain['gen_eta']    = dfTr['tau_visEta']
    dfTrain['gen_phi']    = dfTr['tau_visPhi']
    dfTrain['gen_dm']     = dfTr['tau_DM']

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = dfVal['cl3d_pt']
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['Qcalib_pt']  = valid_Qcalib.ravel()
    dfValid['gen_pt']     = dfVal['tau_visPt']
    dfValid['gen_eta']    = dfVal['tau_visEta']
    dfValid['gen_phi']    = dfVal['tau_visPhi']
    dfValid['gen_dm']     = dfVal['tau_DM']

    inspectWeights(TauQCalibratorModel, 'kernel')
    # inspectWeights(TauQCalibratorModel, 'bias')

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['Qcalib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Quantized Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['Qcalib_pt']/dfTrain['gen_pt']), np.std(dfTrain['Qcalib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['Qcalib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Quantized Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['Qcalib_pt']/dfValid['gen_pt']), np.std(dfValid['Qcalib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['Qcalib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Quantized Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['Qcalib_pt']/dfTrain['gen_pt']), np.std(dfTrain['Qcalib_pt']/dfTrain['gen_pt'])), color='blue',  ls='--', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['Qcalib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Quantized Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['Qcalib_pt']/dfValid['gen_pt']), np.std(dfValid['Qcalib_pt']/dfValid['gen_pt'])), color='green', ls='--', lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.xlim(0., 2.)
    plt.ylim(0., 3.5)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/responses_comparison_quantization.pdf')
    plt.close()

    # PLOTS PER DM
    DMdict = {
            0  : r'$h^{\pm}$',
            1  : r'$h^{\pm}\pi^{0}$',
            10 : r'$h^{\pm}h^{\mp}h^{\pm}$',
            11 : r'$h^{\pm}h^{\mp}h^{\pm}\pi^{0}$',
        }

    tmp0 = dfValid[dfValid['gen_dm']==0]
    tmp1 = dfValid[(dfValid['gen_dm']==1) | (dfValid['gen_dm']==2)]
    tmp10 = dfValid[dfValid['gen_dm']==10]
    tmp11 = dfValid[(dfValid['gen_dm']==11) | (dfValid['gen_dm']==12)]
    plt.figure(figsize=(10,10))
    plt.hist(tmp0['uncalib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['uncalib_pt']/tmp0['gen_pt']), np.std(tmp0['uncalib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['uncalib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['uncalib_pt']/tmp1['gen_pt']), np.std(tmp1['uncalib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['uncalib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['uncalib_pt']/tmp10['gen_pt']), np.std(tmp10['uncalib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['uncalib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['uncalib_pt']/tmp11['gen_pt']), np.std(tmp11['uncalib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(tmp0['Qcalib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['Qcalib_pt']/tmp0['gen_pt']), np.std(tmp0['Qcalib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['Qcalib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['Qcalib_pt']/tmp1['gen_pt']), np.std(tmp1['Qcalib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['Qcalib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['Qcalib_pt']/tmp10['gen_pt']), np.std(tmp10['Qcalib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['Qcalib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['Qcalib_pt']/tmp11['gen_pt']), np.std(tmp11['Qcalib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/calibrated_DM_responses_comparison.pdf')
    plt.close()


    # 2D REPOSNSE VS ETA
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['Qcalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$|\eta^{Gen \tau}|$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/response_vs_eta_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PHI
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['Qcalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$\phi^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/response_vs_phi_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PT
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['Qcalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$p_{T}^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    # plt.xlim(-0.1,5)
    plt.xlim(0.0,2.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/response_vs_pt_comparison.pdf')
    plt.close()


    # L1 TO GEN MAPPING
    dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'],
                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                                   # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                                   labels=False,
                                   include_lowest=True)

    dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'],
                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                                   # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                                   labels=False,
                                   include_lowest=True)

    pt_bins_centers = np.arange(17.5,157.5,5)

    trainL1 = dfTrain.groupby('gen_pt_bin')['Qcalib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['Qcalib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['Qcalib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['Qcalib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['uncalib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauQDNNCalibrator_plots/GenToUncalinL1_pt.pdf')
    plt.close()

# restore normal output
sys.stdout = sys.__stdout__

