from scipy.optimize import curve_fit
from optparse import OptionParser
from array import array
import numpy as np
import pickle
import ROOT
import sys
import os

import matplotlib.pyplot as plt
import matplotlib
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

def sigmoid(x , a, x0, k):
    return a / ( 1 + np.exp(-k*(x-x0)) )


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option("--NtupleV",          dest="NtupleV",        default=None)
    parser.add_option("--v",                dest="v",              default=None)
    parser.add_option("--date",             dest="date",           default=None)
    parser.add_option('--loop',             dest='loop',           default=False, action="store_true")
    parser.add_option('--etaEr',            dest='etaEr',          default=3.0,   type=float)
    parser.add_option('--caloClNxM',        dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",        dest="seedEtCut",      default="2p5")
    parser.add_option("--WP",               dest="WP",             default="99")
    parser.add_option('--CBCEsplit',        dest='CBCEsplit',      default=1.5, type=float)
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/rate'+tag)

    WP = options.WP

    thresholds = np.arange(10.0,160.0,5.0)

    if options.loop:
        
        ## MINE
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/MinBias_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
        inChain = ROOT.TChain("L1CaloTauNtuplizerProducer/L1TauMinatorTree");
        inChain.Add(directory+'/Ntuple_*.root');

        ## EMYR
        # directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV
        # inChain = ROOT.TChain("l1PhaseIITree/L1PhaseIITree");        
        # inChain.Add(directory+'/ntuple_minbias_emyr.root');
        
        ntot = inChain.GetEntries()

        def CaloTauOfflineEtCutBarrel(online) : return (online+1.676)/1.498
        def CaloTauOfflineEtCutEndcap(online) : return (online+6.207)/1.662

        offrate = []
        onlrate = []

        for x in thresholds:

            ## MINE
            offlinescalingcut = "( (abs(minatedl1tau_eta[])<1.5 && minatedl1tau_pt[]>("+str(CaloTauOfflineEtCutBarrel(x))+")) || (abs(minatedl1tau_eta[])>1.5 && minatedl1tau_pt[]>("+str(CaloTauOfflineEtCutEndcap(x))+")) )"
            offlinecut = "Sum$( "+offlinescalingcut+" && abs(minatedl1tau_eta[])<2.4)>0"
            onlinecut  = "Sum$( minatedl1tau_pt[]>"+str(x)+"  && abs(minatedl1tau_eta[])<2.4)>0"

            ## EMYR
            # offlinescalingcut = "( (abs(nnCaloTauEta[])<1.5 && nnCaloTauPt[]>("+str(CaloTauOfflineEtCutBarrel(x))+")) || (abs(nnCaloTauEta[])>1.5 && nnCaloTauPt[]>("+str(CaloTauOfflineEtCutEndcap(x))+")) )"
            # offlinecut = "Sum$( "+offlinescalingcut+" && abs(nnCaloTauEta[])<2.4)>0"
            # onlinecut  = "Sum$( nnCaloTauPt[]>"+str(x)+"  && abs(nnCaloTauEta[])<2.4)>0"

            npass = inChain.GetEntries(offlinecut)
            offrate.append(round(float(npass)/float(ntot)*31038.,1))

            print('SingleTau Offline Rate @ '+str(x)+'GeV :', round(float(npass)/float(ntot)*31038.,1))

            npass = inChain.GetEntries(onlinecut)
            onlrate.append(round(float(npass)/float(ntot)*31038.,1))


        save_obj(offrate, perfdir+'/rate'+tag+'/menutest_offrate.pkl')
        save_obj(onlrate, perfdir+'/rate'+tag+'/menutest_onlrate.pkl')

    else:
        offrate = load_obj(perfdir+'/rate'+tag+'/menutest_offrate.pkl')
        onlrate = load_obj(perfdir+'/rate'+tag+'/menutest_onlrate.pkl')

    plt.figure(figsize=(10,10))
    plt.plot(thresholds, onlrate, linewidth=2, color='black', label=r'Full detector')
    plt.yscale("log")
    plt.ylim(bottom=1)
    legend = plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel('Offline threshold [GeV]')
    plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
    plt.ylim(1,4E4)
    plt.xlim(0,200)
    myThr = np.interp(31, onlrate[::-1], thresholds[::-1])
    myRate = np.interp(150, thresholds, onlrate)
    # plt.hlines(myRate, 0, 150, lw=2, color='dimgray')
    # plt.vlines(150, 1, 200, lw=2, color='dimgray')
    # plt.hlines(31, 0, myThr, lw=2, color='black')
    # plt.vlines(myThr, 1, 200, lw=2, color='black')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/rate'+tag+'/menutest_rate_online_wp'+WP+'.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(thresholds, offrate, linewidth=2, color='black', label=r'Full detector')
    plt.yscale("log")
    plt.ylim(bottom=1)
    legend = plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel('Offline threshold [GeV]')
    plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
    plt.ylim(1,4E4)
    plt.xlim(0,200)
    myThr = np.interp(33, offrate[::-1], thresholds[::-1])
    myRate = np.interp(90, thresholds, offrate)
    # plt.hlines(myRate, 0, 90, lw=2, color='dimgray')
    # plt.vlines(90, 1, 200, lw=2, color='dimgray')
    # plt.hlines(33, 0, myThr, lw=2, color='black')
    # plt.vlines(myThr, 1, 200, lw=2, color='black')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/rate'+tag+'/menutest_rate_offline_wp'+WP+'.pdf')
    plt.close()


