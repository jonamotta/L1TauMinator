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

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTIdentifierTraining'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v
    os.system('mkdir -p '+outdir+'/TauDNNOptimization')

    dfTrIdent = pd.read_pickle('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTIdentifierTraining'+options.inTag+'/X_Ident_BDT_forIdentifier.pkl')
    dfTrCalib = pd.read_pickle('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTCalibratorTraining'+options.inTag+'/X_Calib_BDT_forCalibrator.pkl')

    pt = ['cl3d_pt']
    allFeatures = ['cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ']

    scalerIdent = StandardScaler()
    scaledIdent = pd.DataFrame(scalerIdent.fit_transform(dfTrIdent[allFeatures]), columns=allFeatures)

    scalerCalib = StandardScaler()
    scaledCalib = pd.DataFrame(scalerCalib.fit_transform(dfTrCalib[allFeatures+pt]), columns=allFeatures+pt)

    TrTensorizedTargetIdent = dfTrIdent['targetId'].to_numpy()
    TrTensorizedTargetCalib = dfTrCalib['tau_visPt'].to_numpy()


    ############################# Models definition ##############################

    # This model identifies the tau object:
    #    - one DNN that takes the features of the CL3D clusters

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauDNNOptimization/training.log')
        print(options)

        optHistory = []
        varHistory = []
        availableFeatures = allFeatures

        # train with all possible features
        TrTensorizedInputIdent  = scaledIdent[list(availableFeatures)].to_numpy()
        TrTensorizedInputCalib  = scaledCalib[list(availableFeatures)+pt].to_numpy()

        featuresIdent = keras.Input(shape=len(availableFeatures), name='CL3DFeatures')
        x = featuresIdent
        x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
        TauIdentified = x

        featuresCalib = keras.Input(shape=len(availableFeatures+pt), name='CL3DFeatures')
        x = featuresCalib
        x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        TauCalibrated = x

        TauIdentifierModel = keras.Model(featuresIdent, TauIdentified, name='TauDNNIdentifier')
        TauCalibratorModel = keras.Model(featuresCalib, TauCalibrated, name='TauDNNCalibrator')

        TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(),
                                   run_eagerly=True)

        TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   run_eagerly=True)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        historyIdent = TauIdentifierModel.fit(TrTensorizedInputIdent, TrTensorizedTargetIdent, epochs=400, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)
        historyCalib = TauCalibratorModel.fit(TrTensorizedInputCalib, TrTensorizedTargetCalib, epochs=400, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

        keras.backend.clear_session()

        optHistory.append(min(historyIdent.history['val_loss'])+min(historyCalib.history['val_loss'])/100)
        varHistory.append(list(availableFeatures))

        # train with all possible subsets and permutations of features
        while len(availableFeatures)>1:
            tmp_optHistory = []
            tmp_varHistory = []
            
            print('\n** INFO : Testing training with '+str(len(availableFeatures)-1)+' features')
            for feats in combinations(availableFeatures, r=len(availableFeatures) - 1):
                TrTensorizedInputIdent  = scaledIdent[list(feats)].to_numpy()
                TrTensorizedInputCalib  = scaledCalib[list(feats)+pt].to_numpy()

                featuresIdent = keras.Input(shape=len(feats), name='CL3DFeatures')
                x = featuresIdent
                x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
                x = layers.Activation('relu', name='RELU_DNNlayer')(x)
                x = layers.Dense(1, use_bias=False, name="DNNout")(x)
                x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
                TauIdentified = x

                featuresCalib = keras.Input(shape=len(list(feats)+pt), name='CL3DFeatures')
                x = featuresCalib
                x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
                x = layers.Activation('relu', name='RELU_DNNlayer')(x)
                x = layers.Dense(1, use_bias=False, name="DNNout")(x)
                TauCalibrated = x

                TauIdentifierModel = keras.Model(featuresIdent, TauIdentified, name='TauDNNIdentifier')
                TauCalibratorModel = keras.Model(featuresCalib, TauCalibrated, name='TauDNNCalibrator')

                TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                           loss=tf.keras.losses.BinaryCrossentropy(),
                                           run_eagerly=True)

                TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                           loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                           run_eagerly=True)

                historyIdent = TauIdentifierModel.fit(TrTensorizedInputIdent, TrTensorizedTargetIdent, epochs=400, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)
                historyCalib = TauCalibratorModel.fit(TrTensorizedInputCalib, TrTensorizedTargetCalib, epochs=400, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

                keras.backend.clear_session()

                tmp_optHistory.append(min(historyIdent.history['val_loss'])+min(historyCalib.history['val_loss'])/100)
                tmp_varHistory.append(list(feats))

            minIdx = tmp_optHistory.index(min(tmp_optHistory))
            optHistory.append(tmp_optHistory[minIdx])
            varHistory.append(tmp_varHistory[minIdx])
            availableFeatures = tmp_varHistory[minIdx]

        save_obj(optHistory, outdir+'/TauDNNOptimization/optHistory.pkl')
        save_obj(varHistory, outdir+'/TauDNNOptimization/varHistory.pkl')

    else:
        optHistory = load_obj(outdir+'/TauDNNOptimization/optHistory.pkl')
        varHistory = load_obj(outdir+'/TauDNNOptimization/varHistory.pkl')

    ############################## Optimization validation ##############################

    fig, ax = plt.subplots(figsize=(40,20))
    x = np.linspace(1,len(varHistory),len(varHistory))
    plt.plot(x, optHistory, marker='o', color='black', lw=4, markersize=10)
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
    for i, v in enumerate(optHistory):
        if i==13:   ax.text(i+1, v+0.03, "%.3f"%v, ha="center", fontsize=30)
        elif i==12: ax.text(i+1, v-0.03, "%.3f"%v, ha="center", fontsize=30)
        else:       ax.text(i+1, v-0.015, "%.3f"%v, ha="center", fontsize=30)
    plt.grid(linestyle=':')
    # plt.ylim(min(optHistory)*0.9,max(optHistory)*1.1)
    plt.ylim(max(optHistory)*1.1,min(optHistory)*0.85)
    plt.xlabel('DNN Features', fontsize=50, labelpad=20)
    plt.ylabel(r'Loss$^{Ident.}$ + Loss$^{Calib.}$/100', fontsize=50)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=50)
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, hspace = 0, wspace = 0)
    plt.savefig(outdir+'/TauDNNOptimization/optimizationOPT.pdf')
    plt.close()


# restore normal output
sys.stdout = sys.__stdout__

