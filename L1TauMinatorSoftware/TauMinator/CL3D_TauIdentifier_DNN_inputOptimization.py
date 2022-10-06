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
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNIdentifierTraining'+options.inTag
    os.system('mkdir -p '+outdir+'/TauDNNIdentifierOptimization')

    dfTr = pd.read_pickle(indir+'/X_Ident_BDT_forIdentifier.pkl')
    dfTr['cl3d_abseta'] = abs(dfTr['cl3d_eta']).copy(deep=True)

    allFeatures = ['cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ']

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(dfTr[allFeatures]), columns=allFeatures)

    TrTensorizedTarget = dfTr['targetId'].to_numpy()


    ############################# Models definition ##############################

    # This model identifies the tau object:
    #    - one DNN that takes the features of the CL3D clusters

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauDNNIdentifierOptimization/training.log')
        print(options)

        aucHistory = []
        varHistory = []
        availableFeatures = allFeatures

        # train with all possible features
        TrTensorizedInput  = scaled[list(availableFeatures)].to_numpy()

        features = keras.Input(shape=len(availableFeatures), name='CL3DFeatures')

        x = features
        x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
        TauIdentified = x

        TauIdentifierModel = keras.Model(features, TauIdentified, name='TauDNNIdentifier')

        metrics2follow = [tf.keras.metrics.AUC()]
        TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(),
                                   metrics=metrics2follow,
                                   run_eagerly=True)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauIdentifierModel.fit(TrTensorizedInput, TrTensorizedTarget, epochs=200, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

        keras.backend.clear_session()

        aucHistory.append(max(history.history['val_auc']))
        varHistory.append(list(availableFeatures))

        # train with all possible subsets and permutations of features
        while len(availableFeatures)>1:
            tmp_aucHistory = []
            tmp_varHistory = []
            
            print('\n** INFO : Testing training with '+str(len(availableFeatures)-1)+' features')
            for feats in combinations(availableFeatures, r=len(availableFeatures) - 1):
                TrTensorizedInput  = scaled[list(feats)].to_numpy()

                features = keras.Input(shape=len(feats), name='CL3DFeatures')

                x = features
                x = layers.Dense(16, use_bias=False, name="DNNlayer")(x)
                x = layers.Activation('relu', name='RELU_DNNlayer')(x)
                x = layers.Dense(1, use_bias=False, name="DNNout")(x)
                x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
                TauIdentified = x

                TauIdentifierModel = keras.Model(features, TauIdentified, name='TauDNNIdentifier')

                metrics2follow = [tf.keras.metrics.AUC()]
                TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                           loss=tf.keras.losses.BinaryCrossentropy(),
                                           metrics=metrics2follow,
                                           run_eagerly=True)

                history = TauIdentifierModel.fit(TrTensorizedInput, TrTensorizedTarget, epochs=200, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

                keras.backend.clear_session()

                tmp_aucHistory.append(max(history.history['val_auc']))
                tmp_varHistory.append(list(feats))

            maxIdx = tmp_aucHistory.index(max(tmp_aucHistory))
            aucHistory.append(tmp_aucHistory[maxIdx])
            varHistory.append(tmp_varHistory[maxIdx])
            availableFeatures = tmp_varHistory[maxIdx]

        save_obj(aucHistory, outdir+'/TauDNNIdentifierOptimization/aucHistory.pkl')
        save_obj(varHistory, outdir+'/TauDNNIdentifierOptimization/varHistory.pkl')

    else:
        aucHistory = load_obj(outdir+'/TauDNNIdentifierOptimization/aucHistory.pkl')
        varHistory = load_obj(outdir+'/TauDNNIdentifierOptimization/varHistory.pkl')

    ############################## Optimization validation ##############################

    fig, ax = plt.subplots(figsize=(40,20))
    x = np.linspace(1,len(varHistory),len(varHistory))
    plt.plot(x, aucHistory, marker='o', color='black', lw=4, markersize=10)
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
    for i, v in enumerate(aucHistory):
        if i!=13: ax.text(i+1, v+0.005, "%.3f"%v, ha="center", fontsize=30)
        else:     ax.text(i+1, v-0.01, "%.3f"%v, ha="center", fontsize=30)
    plt.grid(linestyle=':')
    plt.ylim(min(aucHistory)*0.9,1)
    plt.xlabel('DNN Features', fontsize=50, labelpad=20)
    plt.ylabel('AUC', fontsize=50)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=50)
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, hspace = 0, wspace = 0)
    plt.savefig(outdir+'/TauDNNIdentifierOptimization/optimizationAUC.pdf')
    plt.close()


# restore normal output
sys.stdout = sys.__stdout__

