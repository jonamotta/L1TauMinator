from tensorflow.keras.initializers import RandomNormal as RN
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
    parser.add_option("--v",            dest="v",                                default=None)
    parser.add_option("--date",         dest="date",                             default=None)
    parser.add_option("--inTag",        dest="inTag",                            default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
    parser.add_option('--dm_weighted',  dest='dm_weighted', action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get models inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/EmuValidation/TauMinator_CB'
    modelsdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/'

    # X1 = np.load(outdir+'/tensors/images.npz')['arr_0']
    # X2 = np.load(outdir+'/tensors/posits.npz')['arr_0']
    # IDscore_emu  = np.load(outdir+'/tensors/idscore_emu.npz')['arr_0']
    # calibPt_emu  = np.load(outdir+'/tensors/calibpt_emu.npz')['arr_0']

    X1 = np.load('/data_CMS/cms/motta/Phase2L1T/2023_05_24_v15/EmuValidation/HHbbtautau_cltw5x9_seedEtCut2p5/barrel/CLTWimages_5x9_CB_307.npz')['arr_0']
    X2 = np.load('/data_CMS/cms/motta/Phase2L1T/2023_05_24_v15/EmuValidation/HHbbtautau_cltw5x9_seedEtCut2p5/barrel/CLTWpositions_5x9_CB_307.npz')['arr_0']
    IDscore_emu  = np.load('/data_CMS/cms/motta/Phase2L1T/2023_05_24_v15/EmuValidation/HHbbtautau_cltw5x9_seedEtCut2p5/barrel/EMUidscore_5x9_CB_307.npz')['arr_0']
    calibPt_emu  = np.load('/data_CMS/cms/motta/Phase2L1T/2023_05_24_v15/EmuValidation/HHbbtautau_cltw5x9_seedEtCut2p5/barrel/EMUcalibpt_5x9_CB_307.npz')['arr_0']

    # select only emulated that make sense and their counterparts
    sel = calibPt_emu<5000
    X1 = X1[sel]
    X2 = X2[sel]
    IDscore_emu = IDscore_emu[sel]
    calibPt_emu = calibPt_emu[sel]

    # print(X1.shape)
    # print(X2.shape)
    # print(IDscore_emu.shape)
    # print(calibPt_emu.shape)

    CNN = keras.models.load_model(modelsdir+'/TauMinator_CB_cltw5x9_Training/CNNmodel', compile=False)
    DNNid = keras.models.load_model(modelsdir+'/TauMinator_CB_cltw5x9_Training/ID_DNNmodel', compile=False)
    DNNcal = keras.models.load_model(modelsdir+'/TauMinator_CB_cltw5x9_Training/CAL_DNNmodel', compile=False)

    CNNprediction = CNN.predict([X1, X2])
    IDscore_tf = DNNid.predict(CNNprediction)
    calibPt_tf = DNNcal.predict(CNNprediction)


    # print(min(IDscore_tf), max(IDscore_tf))
    # print(min(calibPt_tf), max(calibPt_tf))

    # print(min(IDscore_emu), max(IDscore_emu))
    # print(min(calibPt_emu), max(calibPt_emu))

    for idx in range(len(X1)):
        if calibPt_emu[idx] > 0:
            print('IDscore_emu', IDscore_emu[idx], ' - calibPt_emu', calibPt_emu[idx])
            print('IDscore_tf', IDscore_tf[idx], ' - calibPt_tf', calibPt_tf[idx])
            # print(X1[idx])
            # print(X2[idx])
            print('\n')


    plt.figure(figsize=(10,10))
    plt.hist(IDscore_tf, bins=np.arange(0,1,0.02), label='TF', color='green', density=True, histtype='step', lw=2)
    n,bins,patches = plt.hist(IDscore_emu, bins=np.arange(0,1,0.02), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/CNN_score.pdf')
    plt.close()


    plt.figure(figsize=(10,10))
    plt.hist(calibPt_tf, bins=np.arange(0,200,2), label='TF', color='green', density=True, histtype='step', lw=2)
    n,bins,patches = plt.hist(calibPt_emu, bins=np.arange(0,200,2), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'$p_{T}^{L1, \tau}$')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/calibrated_pt.pdf')
    plt.close()







