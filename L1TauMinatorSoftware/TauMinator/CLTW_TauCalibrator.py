from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import os

np.random.seed(7)

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
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNCalibrator_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 4)
    #       target: visPt, visEta, visPhi, DM

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forCalibrator.npz')['arr_0']


    ############################## Model definition ##############################

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 
    #    - the custom loss targets the visPt of the tau

    # class Custom_LogRectifier(tf.keras.layers.Layer):
    #     def __init__(self, num_outputs):
    #         super(Custom_LogRectifier, self).__init__()
    #         self.num_outputs = num_outputs

    #     def build(self, input_shape):
    #         self.k0 = self.add_weight(shape=[int(input_shape[-1]), self.num_outputs], initializer = 'random_uniform', trainable=True)
    #         self.k1 = self.add_weight(shape=[int(input_shape[-1]), self.num_outputs], initializer = 'random_uniform', trainable=True)
    #         self.k2 = self.add_weight(shape=[int(input_shape[-1]), self.num_outputs], initializer = 'random_uniform', trainable=True)
    #         self.k3 = self.add_weight(shape=[int(input_shape[-1]), self.num_outputs], initializer = 'random_uniform', trainable=True)
    #         self.k4 = self.add_weight(shape=[int(input_shape[-1]), self.num_outputs], initializer = 'random_uniform', trainable=True)

    #     def call(self, inputs):
    #         log = tf.math.log(inputs)
    #         return self.k0 + self.k1*log + self.k2*tf.math.pow(log, 2) + self.k3*tf.math.pow(log, 3) +  self.k4*tf.math.pow(log, 4)

    if options.train:
        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')
        CNN = models.Sequential(name="CNNcalibrator")
        DNN = models.Sequential(name="DNNcalibrator")

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        CNN.add( layers.Conv2D(16, wndw, input_shape=(N, M, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.BatchNormalization(name='BNlayer1') )
        CNN.add( layers.Activation('relu', name='reluCNNlayer1') )
        CNN.add( layers.MaxPooling2D(wndw, name="CNNlayer2") )
        CNN.add( layers.Conv2D(24, wndw, kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.BatchNormalization(name='BNlayer2') )
        CNN.add( layers.Activation('relu', name='reluCNNlayer3') )
        CNN.add( layers.Flatten(name="CNNflatened") )

        DNN.add( layers.Dense(45, name="DNNlayer1") )
        DNN.add( layers.Activation('relu', name='reluDNNlayer1') )
        DNN.add( layers.Dense(16, name="DNNlayer2") )
        DNN.add( layers.Activation('relu', name='reluDNNlayer2') )
        DNN.add( layers.Dense(1, name="DNNout") )
        # DNN.add( Custom_LogRectifier(1) )

        CNNflatened = CNN(layers.Lambda(lambda x : x, name="CNNlayer0")(images))
        middleMan = layers.Concatenate(axis=1, name='middleMan')([CNNflatened, positions])
        TauCalibrated = DNN(layers.Lambda(lambda x : x, name="TauCalibrator")(middleMan))

        TauCalibratorModel = keras.Model([images, positions], TauCalibrated, name='TauCNNCalibrator')

        def custom_loss(y_true, y_pred):
            return tf.nn.l2_loss( (y_true - y_pred) / (y_true + 0.1) )

        # tf.keras.losses.MeanSquaredError()              # loss = square(y_true - y_pred)
        # tf.keras.losses.MeanAbsoluteError()             # loss = abs(y_true - y_pred)
        # tf.keras.losses.MeanAbsolutePercentageError()   # loss = 100 * abs((y_true - y_pred) / y_true)
        # tf.keras.losses.MeanSquaredLogarithmicError()   # loss = square(log(y_true + 1.) - log(y_pred + 1.))

        TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   metrics=['RootMeanSquaredError'],
                                   run_eagerly=True)


        # print(TauCalibrated)
        # print(CNN.summary())
        # print(DNN.summary())
        # print(TauCalibratorModel.summary())
        # exit()

        ############################## Model training ##############################

        history = TauCalibratorModel.fit([X1, X2], Y[:,0].reshape(-1,1), epochs=20, batch_size=1024, verbose=1, validation_split=0.1)

        TauCalibratorModel.save(outdir + '/TauCNNCalibrator')

        for metric in history.history.keys():
            if 'val_' in metric: continue

            plt.plot(history.history[metric], label='Training dataset', lw=2)
            plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            if metric=='loss': plt.legend(loc='upper right')
            else:              plt.legend(loc='lower right')
            mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
            plt.savefig(outdir+'/TauCNNCalibrator_plots/'+metric+'.pdf')
            plt.close()

        w = TauCalibratorModel.layers[2].weights[0].numpy()
        h, b = np.histogram(w, bins=100)
        props = dict(boxstyle='square', facecolor='white', edgecolor='black')
        textstr1 = '\n'.join((r'% of zeros = {}'.format(np.sum(w==0)/np.size(w))))
        plt.figure(figsize=(10,10))
        plt.bar(b[:-1], h, width=b[1]-b[0])
        # plt.text(w.min()+0.01, h.max()-1, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        # plt.legend(loc = 'upper left')
        plt.grid(linestyle=':')
        plt.yscale('log')
        plt.xlabel('Weight value')
        plt.ylabel('Recurrence')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauCNNCalibrator_plots/CNN_sparsity.pdf')
        plt.close()

        w = TauCalibratorModel.layers[6].weights[0].numpy()
        h, b = np.histogram(w, bins=100)
        props = dict(boxstyle='square', facecolor='white', edgecolor='black')
        textstr1 = '\n'.join((r'% of zeros = {}'.format(np.sum(w==0)/np.size(w))))
        plt.figure(figsize=(10,10))
        plt.bar(b[:-1], h, width=b[1]-b[0])
        # plt.text(w.min()+0.01, h.max()-1, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        # plt.legend(loc = 'upper left')
        plt.grid(linestyle=':')
        plt.yscale('log')
        plt.xlabel('Weight value')
        plt.ylabel('Recurrence')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauCNNCalibrator_plots/DNN_sparsity.pdf')
        plt.close()

    else:
        TauCalibratorModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag+'/TauCNNCalibrator', compile=False)


    ############################## Model validation ##############################

    X1_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/Y_Calib_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    train_calib = TauCalibratorModel.predict([X1, X2])
    valid_calib = TauCalibratorModel.predict([X1_valid, X2_valid])
    
    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel()
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,1].ravel()
    dfTrain['gen_phi']    = Y[:,2].ravel()
    dfTrain['gen_dm']     = Y[:,3].ravel()

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,1].ravel()
    dfValid['gen_phi']    = Y_valid[:,2].ravel()
    dfValid['gen_dm']     = Y_valid[:,3].ravel()

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator_plots/responses_comparison.pdf')
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
    plt.savefig(outdir+'/TauCNNCalibrator_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(tmp0['calib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['calib_pt']/tmp0['gen_pt']), np.std(tmp0['calib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['calib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['calib_pt']/tmp1['gen_pt']), np.std(tmp1['calib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['calib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['calib_pt']/tmp10['gen_pt']), np.std(tmp10['calib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['calib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['calib_pt']/tmp11['gen_pt']), np.std(tmp11['calib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator_plots/calibrated_DM_responses_comparison.pdf')
    plt.close()


    # 2D REPOSNSE VS ETA
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$|\eta^{Gen \tau}|$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator_plots/response_vs_eta_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PHI
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$\phi^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator_plots/response_vs_phi_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PT
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$p_{T}^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    # plt.xlim(-0.1,5)
    plt.xlim(0.0,2.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator_plots/response_vs_pt_comparison.pdf')
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

    trainL1 = dfTrain.groupby('gen_pt_bin')['calib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['calib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['calib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['calib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNCalibrator_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['uncalib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNCalibrator_plots/GenToUncalinL1_pt.pdf')
    plt.close()


    ############################## Feature importance ##############################

#    # since we have two inputs we pass a list of inputs to the explainer
#    explainer = shap.GradientExplainer(TauCalibratorModel, [X1, X2])
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
#    plt.savefig(outdir+'/TauCNNCalibrator_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNCalibrator_plots/shap1.pdf')
#    plt.close()
