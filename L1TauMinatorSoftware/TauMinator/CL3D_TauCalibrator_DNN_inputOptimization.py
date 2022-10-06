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
import copy
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

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
    os.system('mkdir -p '+outdir+'/TauDNNCalibratorOptimization')

    dfTr = pd.read_pickle(indir+'/X_Calib_BDT_forCalibrator.pkl')
    dfTr['cl3d_abseta'] = abs(dfTr['cl3d_eta']).copy(deep=True)

    pt = ['cl3d_pt']
    allFeatures = ['cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ']

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(dfTr[allFeatures+pt]), columns=allFeatures+pt)

    TrTensorizedTarget = dfTr['tau_visPt'].to_numpy()

    ############################# Models definition ##############################

    # This model calibrates the tau object:
    #    - one DNN that takes the features of the CL3D clusters

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauDNNCalibratorOptimization/training.log')
        print(options)

        mapeHistory = []
        varHistory = []
        availableFeatures = allFeatures

        # train with all possible features
        TrTensorizedInput  = scaled[availableFeatures+pt].to_numpy()

        features = keras.Input(shape=len(availableFeatures+pt), name='CL3DFeatures')

        x = features
        x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        TauCalibrated = x

        TauCalibratorModel = keras.Model(features, TauCalibrated, name='TauDNNCalibrator')

        TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   run_eagerly=True)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauCalibratorModel.fit(TrTensorizedInput, TrTensorizedTarget, epochs=400, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

        keras.backend.clear_session()

        mapeHistory.append(min(history.history['val_loss']))
        varHistory.append(availableFeatures+pt)

        # train with all possible subsets and permutations of features
        while len(availableFeatures)>1:
            tmp_mapeHistory = []
            tmp_varHistory = []
            
            print('\n** INFO : Testing training with '+str(len(availableFeatures)-1)+' features')
            for feats in combinations(availableFeatures, r=len(availableFeatures) - 1):
                TrTensorizedInput  = scaled[list(feats)+pt].to_numpy()

                features = keras.Input(shape=len(feats)+1, name='CL3DFeatures')

                x = features
                x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
                x = layers.Activation('relu', name='RELU_DNNlayer')(x)
                x = layers.Dense(1, use_bias=False, name="DNNout")(x)
                TauCalibrated = x

                TauCalibratorModel = keras.Model(features, TauCalibrated, name='TauDNNCalibrator')

                TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                           loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                           run_eagerly=True)

                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

                history = TauCalibratorModel.fit(TrTensorizedInput, TrTensorizedTarget, epochs=400, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

                keras.backend.clear_session()

                tmp_mapeHistory.append(min(history.history['val_loss']))
                tmp_varHistory.append(list(feats)+pt)

            minIdx = tmp_mapeHistory.index(min(tmp_mapeHistory))
            mapeHistory.append(tmp_mapeHistory[minIdx])
            varHistory.append(tmp_varHistory[minIdx])
            availableFeatures = copy.deepcopy(tmp_varHistory[minIdx])
            availableFeatures.remove('cl3d_pt')

        save_obj(mapeHistory, outdir+'/TauDNNCalibratorOptimization/mapeHistory.pkl')
        save_obj(varHistory, outdir+'/TauDNNCalibratorOptimization/varHistory.pkl')
        
    else:
        mapeHistory = load_obj(outdir+'/TauDNNCalibratorOptimization/mapeHistory.pkl')
        varHistory = load_obj(outdir+'/TauDNNCalibratorOptimization/varHistory.pkl')


    ############################## Optimization validation ##############################

    for idx in range(len(mapeHistory)):
        mapeHistory[idx] = 1/mapeHistory[idx]

    fig, ax = plt.subplots(figsize=(40,20))
    x = np.linspace(1,len(varHistory),len(varHistory))
    plt.plot(x, mapeHistory, marker='o', color='black', lw=4, markersize=10)
    ax.set_xticks(x)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(varHistory)):
        string = ""
        for j in varHistory[i]:
            string = string+j.replace("cl3d_","")+"\n"
        labels[i] = string
    ax.set_xticklabels(labels, horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    ax.tick_params(axis="x",direction="in", pad=-10)
    ax.tick_params(axis="y",labelsize=30)
    for i, v in enumerate(mapeHistory):
        if i==12:   ax.text(i+1, v-0.00085, "%.4f"%v, ha="center", fontsize=30)
        elif i==13: ax.text(i+1, v-0.00075, "%.4f"%v, ha="center", fontsize=30)
        else:       ax.text(i+1, v+0.0005, "%.4f"%v, ha="center", fontsize=30)
    plt.grid(linestyle=':')
    plt.ylim(min(mapeHistory)*0.9,max(mapeHistory)*1.025)
    plt.xlabel('DNN Features', fontsize=50, labelpad=20)
    plt.ylabel('1 / Mean Absolute Percentage Error', fontsize=50, labelpad=20)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=50)
    plt.subplots_adjust(left = 0.075, bottom = 0.05, right = 0.95, top = 0.95, hspace = 0, wspace = 0)
    plt.savefig(outdir+'/TauDNNCalibratorOptimization/optimizationMAPE.pdf')
    plt.close()

# restore normal output
sys.stdout = sys.__stdout__

