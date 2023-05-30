from tensorflow.keras.initializers import RandomNormal as RN
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from itertools import combinations
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import shap
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
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

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)


def TauIdentifierModelDefinition(N, M, Nshapes, dm_weighted=False):
    wndw = (2,2)
    if N <  5 and M >= 5: wndw = (1,2)
    if N <  5 and M <  5: wndw = (1,1)
    
    images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
    positions = keras.Input(shape = 2, name='TowerClusterPosition')
    cl3dFeats = keras.Input(shape = Nshapes, name='AssociatedCl3dFeatures')
    
    x_ident = layers.Conv2D(4, wndw, input_shape=(N, M, 3), use_bias=False, name="CNNlayer1")(images)
    x_ident = layers.BatchNormalization(name='BN_CNNlayer1')(x_ident)
    x_ident = layers.Activation('relu', name='RELU_CNNlayer1')(x_ident)
    x_ident = layers.MaxPooling2D(wndw, name="MP_CNNlayer1")(x_ident)
    x_ident = layers.Conv2D(8, wndw, use_bias=False, name="CNNlayer2")(x_ident)
    x_ident = layers.BatchNormalization(name='BN_CNNlayer2')(x_ident)
    x_ident = layers.Activation('relu', name='RELU_CNNlayer2')(x_ident)
    x_ident = layers.Flatten(name="CNNflattened")(x_ident)
    x_ident = layers.Concatenate(axis=1, name='middleMan')([x_ident, positions, cl3dFeats])
    
    y_ident = layers.Dense(16, use_bias=False, name="IDlayer1")(x_ident)
    y_ident = layers.Activation('relu', name='RELU_IDlayer1')(y_ident)
    y_ident = layers.Dense(8, use_bias=False, name="IDlayer2")(y_ident)
    y_ident = layers.Activation('relu', name='RELU_IDlayer2')(y_ident)
    y_ident = layers.Dense(1, use_bias=False, name="IDout")(y_ident)
    y_ident = layers.Activation('sigmoid', name='sigmoid_IDout')(y_ident)
    
    TauIdentified = y_ident

    TauIdentifierModel = keras.Model(inputs=[images, positions, cl3dFeats], outputs=TauIdentified, name='TauMinator_CE_indent')

    if dm_weighted:
        TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                   metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                                   sample_weight_mode='sample-wise',
                                   run_eagerly=True)
    else:
        TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                   metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                                   run_eagerly=True)

    return TauIdentifierModel


def TauCalibratorModelDefinition(Nshapes, dm_weighted=False):
    middleMan = keras.Input(shape=26+Nshapes, name='middleMan')
    
    x_calib = layers.Dense(16, use_bias=False, name="DNNlayer1")(middleMan)
    x_calib = layers.Activation('relu', name='RELU_DNNlayer1')(x_calib)
    x_calib = layers.Dense(8, use_bias=False, name="DNNlayer2")(x_calib)
    x_calib = layers.Activation('relu', name='RELU_DNNlayer2')(x_calib)
    x_calib = layers.Dense(1, use_bias=False, name="DNNout")(x_calib)
    
    TauCalibrated = x_calib

    TauCalibratorModel = keras.Model(inputs=middleMan, outputs=TauCalibrated, name='TauMinator_CE_calib')

    if dm_weighted:
        TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   metrics=['RootMeanSquaredError'],
                                   sample_weight_mode='sample-wise',
                                   run_eagerly=True)

    else:
        TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   metrics=['RootMeanSquaredError'],
                                   run_eagerly=True)

    return TauCalibratorModel

def ExtractCNN(FullTauIdentifierModel):
    image_in = FullTauIdentifierModel.get_layer(index=0).get_output_at(0)
    posit_in = FullTauIdentifierModel.get_layer(name='TowerClusterPosition').get_output_at(0)
    cl3ds_in = FullTauIdentifierModel.get_layer(name='AssociatedCl3dFeatures').get_output_at(0)
    flat_out = FullTauIdentifierModel.get_layer(name='middleMan').get_output_at(0)
    return tf.keras.Model([image_in, posit_in, cl3ds_in], flat_out)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                default=None)
    parser.add_option("--date",         dest="date",                             default=None)
    parser.add_option("--inTag",        dest="inTag",                            default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
    parser.add_option('--train',        dest='train',       action='store_true', default=False)
    parser.add_option('--dm_weighted',  dest='dm_weighted', action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Optimisation'+options.inTag
    if options.dm_weighted: outdir += '_dmWeighted' 
    os.system('mkdir -p '+outdir)

    if options.train:
        # X1 is (None, N, M, 3)
        #       N runs over eta, M runs over phi
        #       3 features are: EgIet, Iem, Ihad
        # 
        # X2 is (None, 2)
        #       2 are eta and phi values
        # 
        # X3 is (None, 17)
        #       12 are the shape variables of the cl3ds
        #
        # Y is (None, 1)
        #       target: particel ID (tau = 1, non-tau = 0)

        X1 = np.load(outdir+'/tensors/images_train.npz')['arr_0']
        X2 = np.load(outdir+'/tensors/posits_train.npz')['arr_0']
        X3 = np.load(outdir+'/tensors/shapes_train.npz')['arr_0']
        Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
        Yid  = Y[:,1].reshape(-1,1)
        Ycal = Y[:,0].reshape(-1,1)

        # scale features of the cl3ds
        scaler = load_obj(indir+'/CL3D_features_scaler/cl3d_features_scaler.pkl')
        X3 = scaler.transform(X3)

        # select only taus for the calibration
        tau_sel = Yid.reshape(1,-1)[0] > 0
        X1_calib = X1[tau_sel]
        X2_calib = X2[tau_sel]
        X3_calib = X3[tau_sel]
        Y_calib = Y[tau_sel]
        Ycal = Ycal[tau_sel]

        if options.dm_weighted:
            # create dm weights for identification
            bkg_sel = Yid.reshape(1,-1)[0] < 1
            Ydm = Y[:,4].reshape(-1,1)
            dm0 = Ydm.reshape(1,-1)[0] == 0
            dm1 = (Ydm.reshape(1,-1)[0] == 1) | (Ydm.reshape(1,-1)[0] == 2)
            dm10 = Ydm.reshape(1,-1)[0] == 10
            dm11 = (Ydm.reshape(1,-1)[0] == 11) | (Ydm.reshape(1,-1)[0] == 12)

            bkg_w = Yid.shape[0] / (Yid.shape[0] - (np.sum(dm0)+np.sum(dm1)+np.sum(dm10)+np.sum(dm11)))
            dm0_w = Yid.shape[0] / np.sum(dm0)
            dm1_w = Yid.shape[0] / np.sum(dm1)
            dm10_w = Yid.shape[0] / np.sum(dm10)
            dm11_w = Yid.shape[0] / np.sum(dm11)

            dm_weights_ident = bkg_w*bkg_sel + dm0_w*dm0 + dm1_w*dm1 + dm10_w*dm10 + dm11_w*dm11

            # create dm weights for calibration
            Ydm = Y[:,4].reshape(-1,1)
            dm0 = Ydm.reshape(1,-1)[0] == 0
            dm1 = (Ydm.reshape(1,-1)[0] == 1) | (Ydm.reshape(1,-1)[0] == 2)
            dm10 = Ydm.reshape(1,-1)[0] == 10
            dm11 = (Ydm.reshape(1,-1)[0] == 11) | (Ydm.reshape(1,-1)[0] == 12)

            dm0_w = np.sum(tau_sel) / np.sum(dm0)
            dm1_w = np.sum(tau_sel) / np.sum(dm1)
            dm10_w = np.sum(tau_sel) / np.sum(dm10)
            dm11_w = np.sum(tau_sel) / np.sum(dm11)

            dm_weights_calib = dm0_w*dm0 + dm1_w*dm1 + dm10_w*dm10 + dm11_w*dm11


        ############################## Models definition ##############################

        # This model identifies the tau object:
        #    - one CNN that takes eg, em, had deposit images
        #    - one DNN that takes the flat output of the the CNN and the cluster position and the cl3ds features

        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/training.log')
        print(options)

        allFeatures = np.array(['cl3d_pt', 'cl3d_e', 'cl3d_eta', 'cl3d_phi', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ'])
        allIndexes  = np.array([0        , 1       , 2         , 3         , 4                  , 5                      , 6                , 7            , 8            , 9            , 10           , 11        , 12           , 13           , 14            , 15        , 16                  ])

        optHistory = []
        optHistory_ident = []
        optHistory_calib = []
        varHistory = []
        availableFeatures = allFeatures
        availableIndexes  = allIndexes

        TauIdentifierModel = TauIdentifierModelDefinition(N, M, len(allFeatures), options.dm_weighted)
        TauCalibratorModel = TauCalibratorModelDefinition(len(allFeatures), options.dm_weighted)

        # print(TauIdentifierModel.summary())
        # print(TauCalibratorModel.summary())
        # exit()

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]


        ############################## Identifier model training ##############################

        if options.dm_weighted: history_identifier = TauIdentifierModel.fit([X1, X2, X3[:,list(allIndexes)]], Yid, epochs=200, batch_size=2048, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=dm_weights_ident)
        else:                   history_identifier = TauIdentifierModel.fit([X1, X2, X3[:,list(allIndexes)]], Yid, epochs=200, batch_size=2048, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks)

        # split convolutional part alone
        CNNmodel = ExtractCNN(TauIdentifierModel)

        # print(CNNmodel.summary())
        # exit()

        keras.backend.clear_session()

        ############################## Calibrator model training ##############################

        CNNprediction = CNNmodel([X1_calib, X2_calib, X3_calib[:,list(allIndexes)]], training=False)

        if options.dm_weighted: history_calibrator = TauCalibratorModel.fit(CNNprediction, Ycal, epochs=200, batch_size=2048, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=dm_weights_calib)
        else:                   history_calibrator = TauCalibratorModel.fit(CNNprediction, Ycal, epochs=200, batch_size=2048, verbose=1, validation_split=0.25, callbacks=callbacks)

        keras.backend.clear_session()

        optHistory.append(min(history_identifier.history['val_loss'])+min(history_calibrator.history['val_loss'])/100)
        optHistory_ident.append(min(history_identifier.history['val_loss']))
        optHistory_calib.append(min(history_calibrator.history['val_loss'])/100)
        varHistory.append(list(availableFeatures))


        # train with all possible subsets and permutations of features
        while len(availableIndexes)>1:
            tmp_optHistory = []
            tmp_optHistory_ident = []
            tmp_optHistory_calib = []
            tmp_varHistory = []

            print('\n** INFO : Testing training with '+str(len(availableIndexes)-1)+' features')
            for feats in combinations(availableIndexes, r=len(availableIndexes) - 1):
                
                training_X3       = X3[:,list(feats)]
                training_X3_calib = X3_calib[:,list(feats)]

                TauIdentifierModel = TauIdentifierModelDefinition(N, M, len(feats), options.dm_weighted)
                TauCalibratorModel = TauCalibratorModelDefinition(len(feats), options.dm_weighted)

                ############################## Identifier model training ##############################

                if options.dm_weighted: history_identifier = TauIdentifierModel.fit([X1, X2, training_X3], Yid, epochs=200, batch_size=2048, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=dm_weights_ident)
                else:                   history_identifier = TauIdentifierModel.fit([X1, X2, training_X3], Yid, epochs=200, batch_size=2048, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks)

                # split convolutional part alone
                CNNmodel = ExtractCNN(TauIdentifierModel)

                keras.backend.clear_session()

                ############################## Calibrator model training ##############################

                CNNprediction = CNNmodel([X1_calib, X2_calib, training_X3_calib], training=False)

                if options.dm_weighted: history_calibrator = TauCalibratorModel.fit(CNNprediction, Ycal, epochs=200, batch_size=2048, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=dm_weights_calib)
                else:                   history_calibrator = TauCalibratorModel.fit(CNNprediction, Ycal, epochs=200, batch_size=2048, verbose=1, validation_split=0.25, callbacks=callbacks)

                keras.backend.clear_session()

                tmp_optHistory.append(min(history_identifier.history['val_loss'])+min(history_calibrator.history['val_loss'])/100)
                tmp_optHistory_ident.append(min(history_identifier.history['val_loss']))
                tmp_optHistory_calib.append(min(history_calibrator.history['val_loss'])/100)
                tmp_varHistory.append(list(feats))


            # find and store minimum of combinations
            minIdx = tmp_optHistory.index(min(tmp_optHistory))
            optHistory.append(tmp_optHistory[minIdx])
            optHistory_ident.append(tmp_optHistory_ident[minIdx])
            optHistory_calib.append(tmp_optHistory_calib[minIdx])
            varHistory.append(list(allFeatures[np.sort(tmp_varHistory[minIdx])]))
            availableIndexes = tmp_varHistory[minIdx]


        save_obj(optHistory, outdir+'/optHistory.pkl')
        save_obj(optHistory_ident, outdir+'/optHistory_ident.pkl')
        save_obj(optHistory_calib, outdir+'/optHistory_calib.pkl')
        save_obj(varHistory, outdir+'/varHistory.pkl')

        # restore normal output
        sys.stdout = sys.__stdout__

    else:
        optHistory = load_obj(outdir+'/optHistory.pkl')
        optHistory_ident = load_obj(outdir+'/optHistory_ident.pkl')
        optHistory_calib = load_obj(outdir+'/optHistory_calib.pkl')
        varHistory = load_obj(outdir+'/varHistory.pkl')


    ############################## Optimization validation ##############################

    allFeatures = np.array(['cl3d_pt', 'cl3d_e', 'cl3d_eta', 'cl3d_phi', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ'])
    allFeatures_nice = np.array([r'$p_T$', r'E', r'$\eta$', r'$\phi$', 'Shower length', 'Core shower length', 'First layer', r'$\sigma_{\eta\eta}^{tot}$', r'$\sigma_{\eta\eta}^{max}$', r'$\sigma_{\phi\phi}^{tot}$', r'$\sigma_{\phi\phi}^{max}$', r'$\sigma_{zz}$', r'$\sigma_{rr}^{tot}$', r'$\sigma_{rr}^{max}$', r'$\sigma_{rr}^{mean}$', 'H/E', '<z>'])

    fig, ax = plt.subplots(figsize=(50,20))
    x = np.linspace(1,len(varHistory),len(varHistory))
    plt.plot(x, optHistory,       marker='o', color='black', lw=4, markersize=10, label=r'Loss$^{Ident.}$ + Loss$^{Calib.}$/100')
    plt.plot(x, optHistory_ident, marker='o', color='red',   lw=4, markersize=10, label=r'Loss$^{Ident.}$')
    plt.plot(x, optHistory_calib, marker='o', color='blue',  lw=4, markersize=10, label=r'Loss$^{Calib.}$/100')
    ax.set_xticks(x)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(varHistory)):
        string = ""
        for var in varHistory[i]:
            # print(var)
            idx = np.where(allFeatures == var)[0][0]
            string += allFeatures_nice[idx]+'\n'
        labels[i] = string
    ax.set_xticklabels(labels, horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    ax.tick_params(axis="x",direction="in", pad=-10)
    ax.tick_params(axis="y",labelsize=30)
    for i, v in enumerate(optHistory):
        ax.text(i+1, v-0.015, "%.3f"%v, ha="center", color='black', fontsize=30)
    for i, v in enumerate(optHistory_ident):
        ax.text(i+1, v-0.015, "%.3f"%v, ha="center", color='red', fontsize=30)
    for i, v in enumerate(optHistory_calib):
        ax.text(i+1, v-0.015, "%.3f"%v, ha="center", color='blue', fontsize=30)
    plt.grid(linestyle=':')
    # plt.ylim(min(optHistory)*0.9,max(optHistory)*1.1)
    ymax = max(max(max(optHistory),max(optHistory_ident)), max(optHistory_calib))*1.25
    ymin = min(min(min(optHistory),min(optHistory_ident)), min(optHistory_calib))*0.80
    plt.ylim(ymax,ymin)
    plt.xlabel('DNN Features', fontsize=50, labelpad=20)
    plt.ylabel(r'Loss', fontsize=50)
    plt.legend(loc=(0.75,0.5), fontsize=40)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=50)
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, hspace = 0, wspace = 0)
    plt.savefig(outdir+'/optimizationOPT_splitLoss.pdf')
    plt.close()



    fig, ax = plt.subplots(figsize=(50,20))
    x = np.linspace(1,len(varHistory),len(varHistory))
    plt.plot(x, optHistory,       marker='o', color='black', lw=4, markersize=10)
    ax.set_xticks(x)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(varHistory)):
        string = ""
        for var in varHistory[i]:
            # print(var)
            idx = np.where(allFeatures == var)[0][0]
            string += allFeatures_nice[idx]+'\n'
        labels[i] = string
    ax.set_xticklabels(labels, horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    ax.tick_params(axis="x",direction="in", pad=-10)
    ax.tick_params(axis="y",labelsize=30)
    for i, v in enumerate(optHistory):
        if i==13:   ax.text(i+1, v+0.03,  "%.3f"%v, ha="center", fontsize=30)
        elif i==12: ax.text(i+1, v-0.03,  "%.3f"%v, ha="center", fontsize=30)
        else:       ax.text(i+1, v-0.015, "%.3f"%v, ha="center", fontsize=30)
    plt.grid(linestyle=':')
    plt.ylim(max(optHistory)*1.1,min(optHistory)*0.9)
    plt.xlabel('DNN Features', fontsize=50, labelpad=20)
    plt.ylabel(r'Loss$^{Ident.}$ + Loss$^{Calib.}$/100', fontsize=50)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=50)
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, hspace = 0, wspace = 0)
    plt.savefig(outdir+'/optimizationOPT_totalLoss.pdf')
    plt.close()



