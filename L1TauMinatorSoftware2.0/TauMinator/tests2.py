from tensorflow.keras.initializers import RandomNormal as RN
from keras_preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cmsml
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

from scipy.optimize import curve_fit
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
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/modelSparsity'+which+'.pdf')
    plt.close()

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    with tf.device('/CPU:0'):
        parser = OptionParser()
        parser.add_option("--v",            dest="v",                                default=None)
        parser.add_option("--date",         dest="date",                             default=None)
        parser.add_option("--inTag",        dest="inTag",                            default="")
        parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
        parser.add_option('--train',        dest='train',       action='store_true', default=False)
        parser.add_option('--dm_weighted',  dest='dm_weighted', action='store_true', default=False)
        parser.add_option('--pt_weighted',  dest='pt_weighted', action='store_true', default=False)
        (options, args) = parser.parse_args()
        print(options)

        # get clusters' shape dimensions
        N = int(options.caloClNxM.split('x')[0])
        M = int(options.caloClNxM.split('x')[1])

        ############################## Get model inputs ##############################

        user = os.getcwd().split('/')[5]
        indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
        outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training'+options.inTag
        tag = ''
        if options.dm_weighted: tag = '_dmWeighted'
        if options.pt_weighted: tag = '_ptWeighted'
        outdir += tag
        os.system('mkdir -p '+outdir+'/TauMinator_CB_calib_plots')
        os.system('mkdir -p '+indir+'/CMSSWmodels'+tag)

        # X1 is (None, N, M, 3)
        #       N runs over eta, M runs over phi
        #       3 features are: EgIet, Iem, Ihad
        # 
        # X2 is (None, 2)
        #       2 are eta and phi values
        #
        # Y is (None, 1)
        #       target: particel ID (tau = 1, non-tau = 0)

        X1 = np.load(outdir+'/tensors/images_train.npz')['arr_0']
        X2 = np.load(outdir+'/tensors/posits_train.npz')['arr_0']
        Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
        Yid  = Y[:,1].reshape(-1,1)
        Ycal = Y[:,0].reshape(-1,1)

        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)
        Ycal_valid = Y_valid[:,0].reshape(-1,1)

        ################################################################################
        # PREPROCESSING OF THE INPUTS

        # merge EG and EM in one single filter
        X1_new = np.zeros((len(X1), 5, 9, 2))
        X1_new[:,:,:,0] = (X1[:,:,:,0] + X1[:,:,:,1])
        X1_new[:,:,:,1] = X1[:,:,:,2]
        X1 = X1_new

        # select only taus
        tau_sel = Yid.reshape(1,-1)[0] > 0
        X1 = X1[tau_sel]
        X2 = X2[tau_sel]
        Y = Y[tau_sel]
        Ycal = Ycal[tau_sel]

        # select only objects below 150GeV
        pt_sel = Ycal.reshape(1,-1)[0] < 150
        # pt_sel = (Ycal.reshape(1,-1)[0] > 40) & (Ycal.reshape(1,-1)[0] < 60)
        X1 = X1[pt_sel]
        X2 = X2[pt_sel]
        Y = Y[pt_sel]
        Ycal = Ycal[pt_sel]

        # select not too weird reconstruuctions
        X1_totE = np.sum(X1, (1,2,3)).reshape(-1,1)
        Yfactor = Ycal / X1_totE
        farctor_sel = (Yfactor.reshape(1,-1)[0] < 2.4) & (Yfactor.reshape(1,-1)[0] > 0.4)
        X1 = X1[farctor_sel]
        X2 = X2[farctor_sel]
        Y = Y[farctor_sel]
        Ycal = Ycal[farctor_sel]
        Yfactor = Yfactor[farctor_sel]

        # normalize everything
        X1 = X1 / 256.
        X2 = X2 / np.pi
        Ycal = Ycal / 256.
        rawPt = np.sum(X1, (1,2,3))

        # make X1 tensor the good shape
        X1 = np.sum(X1, (2,3))
        # X1 = X1.reshape(len(X1),-1)

        # print(min(Ycal))
        # print(max(Ycal))
        # print(np.mean(Ycal))
        # print('-------')
        # print(min(rawPt))
        # print(max(rawPt))
        # print(np.mean(rawPt))
        # print('-------')
        # print(X1.shape)
        # print(X2.shape)
        # print(Y.shape)
        # print(Yid.shape)
        # print(Ycal.shape)
        # print('-------')
        # print('-------')

        ################################################################################
        # PREPROCESSING OF THE VAALIDATION

        # merge EG and EM in one single filter
        X1_valid_new = np.zeros((len(X1_valid), 5, 9, 2))
        X1_valid_new[:,:,:,0] = (X1_valid[:,:,:,0] + X1_valid[:,:,:,1])
        X1_valid_new[:,:,:,1] = X1_valid[:,:,:,2]
        X1_valid = X1_valid_new

        # select only taus
        tau_sel_valid = Yid_valid.reshape(1,-1)[0] > 0
        X1_valid = X1_valid[tau_sel_valid]
        X2_valid = X2_valid[tau_sel_valid]
        Y_valid = Y_valid[tau_sel_valid]
        Ycal_valid = Ycal_valid[tau_sel_valid]

        # select only objects below 150GeV
        pt_sel_valid = Ycal_valid.reshape(1,-1)[0] < 150
        # pt_sel_valid = (Ycal_valid.reshape(1,-1)[0] > 40) & (Ycal_valid.reshape(1,-1)[0] < 60)
        X1_valid = X1_valid[pt_sel_valid]
        X2_valid = X2_valid[pt_sel_valid]
        Y_valid = Y_valid[pt_sel_valid]
        Ycal_valid = Ycal_valid[pt_sel_valid]

        # select not too weird reconstruuctions
        X1_totE_valid = np.sum(X1_valid, (1,2,3)).reshape(-1,1)
        Yfactor_valid = Ycal_valid / X1_totE_valid
        farctor_sel_valid = (Yfactor_valid.reshape(1,-1)[0] < 2.4) & (Yfactor_valid.reshape(1,-1)[0] > 0.4)
        X1_valid = X1_valid[farctor_sel_valid]
        X2_valid = X2_valid[farctor_sel_valid]
        Y_valid = Y_valid[farctor_sel_valid]
        Ycal_valid = Ycal_valid[farctor_sel_valid]
        Yfactor_valid = Yfactor_valid[farctor_sel_valid]

        # normalize everything
        X1_valid = X1_valid / 256.
        X2_valid = X2_valid / np.pi
        Ycal_valid = Ycal_valid / 256.
        rawPt_valid = np.sum(X1_valid, (1,2,3))

        # make X1 tensor the good shape
        X1_valid = np.sum(X1_valid, (2,3))
        # X1_valid = X1_valid.reshape(len(X1_valid),-1)

        # print(min(Ycal_valid))
        # print(max(Ycal_valid))
        # print(np.mean(Ycal_valid))
        # print('-------')
        # print(min(rawPt_valid))
        # print(max(rawPt_valid))
        # print(np.mean(rawPt_valid))
        # print('-------')
        # print(X1_valid.shape)
        # print(X2_valid.shape)
        # print(Y_valid.shape)
        # print(Yid_valid.shape)
        # print(Ycal_valid.shape)

        ################################################################################
        # WEIGHTS CALCULATION

        dfweights = pd.DataFrame(Y[:,0], columns=['gen_pt'])
        # dfweights = pd.DataFrame(np.sum(X1, (1)).ravel()*256., columns=['gen_pt'])
        # weight_Ebins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 130, 150, 1000]
        weight_Ebins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 130, 150]
        dfweights['gen_pt_bin'] = pd.cut(dfweights['gen_pt'], bins=weight_Ebins, labels=False, include_lowest=True)
        dfweights['weight'] = dfweights.shape[0] / dfweights.groupby(['gen_pt_bin'])['gen_pt_bin'].transform('count')
        # dfweights['weight'] = dfweights.apply(lambda row: customPtDownWeight(row) , axis=1)
        dfweights['weight'] = dfweights['weight'] / np.mean(dfweights['weight'])
        pt_weights = dfweights['weight'].to_numpy()

        plt.figure(figsize=(10,10))
        plt.hist(dfweights['gen_pt'], bins=weight_Ebins,                               label="Un-weighted", color='red',   lw=2, histtype='step', alpha=0.7)
        plt.hist(dfweights['gen_pt'], bins=weight_Ebins, weights=dfweights['weight'],  label="Weighted",    color='green', lw=2, histtype='step', alpha=0.7)
        plt.xlabel(r'$p_{T}^{Gen \tau}$')
        plt.ylabel(r'a.u.')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        plt.yscale('log')
        # plt.xlim(0,175)
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig('./tests2/pt_gentau.pdf')
        plt.close()

        plt.figure(figsize=(10,10))
        plt.hist(np.sum(X1, (1)).ravel()*256., bins=weight_Ebins,                               label="Un-weighted", color='red',   lw=2, histtype='step', alpha=0.7)
        plt.hist(np.sum(X1, (1)).ravel()*256., bins=weight_Ebins, weights=dfweights['weight'],  label="Weighted",    color='green', lw=2, histtype='step', alpha=0.7)
        plt.xlabel(r'$p_{T}^{Gen \tau}$')
        plt.ylabel(r'a.u.')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        plt.yscale('log')
        plt.xlim(0,175)
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig('./tests2/pt_l1_uncalib.pdf')
        plt.close()

        ################################################################################
        # NEURAL NETWORK TRAINING

        if options.train:
            images = keras.Input(shape = 5, name='TowerClusterImage')
            positions = keras.Input(shape = 2, name='TowerClusterPosition')
            energy = keras.Input(shape = 1, name='TowerClusterEnergy')

            x = layers.Concatenate(axis=1, name='middleMan')([images, positions, energy])
            y = layers.Dense(180, use_bias=False, name="CALlayer1", kernel_initializer=RN(seed=7))(x)
            y = layers.Activation('relu', name='RELU_CALlayer1')(y)
            y = layers.Dense(75, use_bias=False, name="CALlayer2", kernel_initializer=RN(seed=7))(y)
            y = layers.Activation('relu', name='RELU_CALlayer2')(y)
            y = layers.Dense(75, use_bias=False, name="CALlayer3", kernel_initializer=RN(seed=7))(y)
            y = layers.Activation('relu', name='RELU_CALlayer3')(y)
            y = layers.Dense(75, use_bias=False, name="CALlayer3p", kernel_initializer=RN(seed=7))(y)
            y = layers.Activation('relu', name='RELU_CALlayer3p')(y)
            y = layers.Dense(75, use_bias=False, name="CALlayer3s", kernel_initializer=RN(seed=7))(y)
            y = layers.Activation('relu', name='RELU_CALlayer3s')(y)
            y = layers.Dense(75, use_bias=False, name="CALlayer3t", kernel_initializer=RN(seed=7))(y)
            y = layers.Activation('relu', name='RELU_CALlayer3t')(y)
            y = layers.Dense(1, use_bias=False, name="CALout", kernel_initializer=RN(seed=7))(y)
            y = layers.Activation('sigmoid', name='LIN_CALlayer3')(y)
            TauCalibrated = y

            TauMinatorModel = keras.Model(inputs=[images, positions, energy], outputs=TauCalibrated, name='TauMinator_CB_calib')

            TauMinatorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-7, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.MeanSquaredError(),
                                    metrics=['RootMeanSquaredError'],
                                    # sample_weight_mode='sample-wise',
                                    run_eagerly=True)
            
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)]

            history = TauMinatorModel.fit([X1, X2, rawPt], Ycal, epochs=150, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks, sample_weight=pt_weights)

            TauMinatorModel.save('./tests2/model', include_optimizer=False)

            for metric in history.history.keys():
                    if metric == 'lr':
                        plt.plot(history.history[metric], lw=2)
                        plt.ylabel('Learning rate')
                        plt.xlabel('Epoch')
                        plt.yscale('log')
                        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                        plt.savefig('./tests2/'+metric+'.pdf')
                        plt.close()

                    else:
                        if 'val_' in metric: continue

                        plt.plot(history.history[metric], label='Training dataset', lw=2)
                        plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
                        plt.xlabel('Epoch')
                        if 'loss' in metric:
                            plt.ylabel('Loss')
                            plt.legend(loc='upper right')
                        elif 'auc' in metric:
                            plt.ylabel('AUC')
                            plt.legend(loc='lower right')
                        elif 'binary_accuracy' in metric:
                            plt.ylabel('Binary accuracy')
                            plt.legend(loc='lower right')
                        elif 'root_mean_squared_error' in metric:
                            plt.ylabel('Root Mean Squared Error')
                            plt.legend(loc='upper right')
                        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                        plt.savefig('./tests2/'+metric+'.pdf')
                        plt.close()

        else:
            TauMinatorModel = keras.models.load_model('./tests2/model', compile=False)


        train_calib = TauMinatorModel.predict([X1, X2, rawPt]) * 256.
        valid_calib = TauMinatorModel.predict([X1_valid, X2_valid, rawPt_valid]) * 256.

        train_calib = train_calib.reshape(len(train_calib))
        valid_calib = valid_calib.reshape(len(valid_calib))

        train_plot = Ycal.reshape(len(Ycal)) * 256.
        valid_plot = Ycal_valid.reshape(len(Ycal_valid)) * 256.

        plt.figure(figsize=(10,10))
        plt.hist(train_plot,  label=r'Train. target',    color='red',  ls='-', lw=2, density=True, histtype='step', alpha=0.7, bins=np.arange(18,151,3))
        plt.hist(valid_plot,  label=r'Valid. target',    color='blue', ls='-', lw=2, density=True, histtype='step', alpha=0.7, bins=np.arange(18,151,3))
        plt.hist(train_calib, label=r'Train. predicted', color='red',  ls='--', lw=2, density=True, histtype='step', alpha=0.7, bins=np.arange(18,151,3))
        plt.hist(valid_calib, label=r'Valid. predicted', color='blue', ls='--', lw=2, density=True, histtype='step', alpha=0.7, bins=np.arange(18,151,3))
        plt.xlabel(r'$p_{T}^{L1 \tau}$')
        plt.ylabel(r'a.u.')
        plt.yscale('log')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig('./tests2/responses_comparison.pdf')
        plt.close()


        dfTrain = pd.DataFrame()
        dfTrain['uncalib_pt'] = np.sum(X1, (1)).ravel() * 256.
        dfTrain['calib_pt']   = train_calib.ravel()
        dfTrain['gen_pt']     = Ycal.ravel() * 256.

        dfValid = pd.DataFrame()
        dfValid['uncalib_pt'] = np.sum(X1_valid, (1)).ravel() * 256.
        dfValid['calib_pt']   = valid_calib.ravel()
        dfValid['gen_pt']     = Ycal_valid.ravel() * 256.

        low_edge = 18
        hig_edge = 150
        dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'], bins=np.arange(low_edge,hig_edge+2,3), labels=False, include_lowest=True)
        dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'], bins=np.arange(low_edge,hig_edge+2,3), labels=False, include_lowest=True)
        pt_bins_centers = np.array(np.arange(low_edge+1.5,hig_edge,3))

        uncalib_trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean())
        uncalib_validL1 = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean())
        uncalib_trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std())
        uncalib_validL1std = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].std())

        calib_trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].mean())
        calib_validL1 = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].mean())
        calib_trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].std())
        calib_validL1std = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].std())

        plt.figure(figsize=(10,10))
        plt.errorbar(pt_bins_centers, uncalib_trainL1, yerr=uncalib_trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
        plt.errorbar(pt_bins_centers, uncalib_validL1, yerr=uncalib_validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
        plt.plot(np.array(np.arange(0,200,1)), np.array(np.arange(0,200,1)), label='Ideal calibration', color='black', ls='--', lw=2)
        plt.legend(loc = 'lower right', fontsize=16)
        plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
        plt.xlabel(r'Gen $p_{T}$ [GeV]')
        plt.xlim(17, 151)
        plt.ylim(17, 151)
        plt.grid()
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig('./tests2/GenToUncalibL1_pt.pdf')
        plt.close()


        plt.figure(figsize=(10,10))
        plt.errorbar(pt_bins_centers, calib_trainL1, yerr=calib_trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
        plt.errorbar(pt_bins_centers, calib_validL1, yerr=calib_validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
        plt.plot(np.array(np.arange(0,200,1)), np.array(np.arange(0,200,1)), label='Ideal calibration', color='black', ls='--', lw=2)
        plt.legend(loc = 'lower right', fontsize=16)
        plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
        plt.xlabel(r'Gen $p_{T}$ [GeV]')
        plt.xlim(17, 151)
        plt.ylim(17, 151)
        plt.grid()
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig('./tests2/GenToCalibL1_pt.pdf')
        plt.close()





