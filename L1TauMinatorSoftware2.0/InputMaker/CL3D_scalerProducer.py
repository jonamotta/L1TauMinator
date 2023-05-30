from sklearn.preprocessing import StandardScaler
from optparse import OptionParser
from array import array
import pandas as pd
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

def deltaPhi(phi1, phi2):
    dPhi = abs(phi1 - phi2)
    if dPhi  < np.pi: return dPhi
    else:             return dPhi - 2*np.pi

def inputQuantizer(inputF, LSB, nbits):
    if LSB: return min( np.floor(inputF/LSB), 2**nbits-1 ) * LSB
    else:   return inputF
inputQuantizer_vctd = np.vectorize(inputQuantizer)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option('--pathIn',  dest='pathIn',  default=None)
    parser.add_option('--pathOut', dest='pathOut', default=None)
    (options, args) = parser.parse_args()
    print(options)

    # loop over the events to fill all the tesors
    print('creating chain')
    inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
    inChain.Add(options.pathIn+'/Ntuple_*.root')

    nEntries = inChain.GetEntries()
    if nEntries == 0: exit()

    featDict = {'cl3d_pt' : [],
                'cl3d_energy' : [],
                'cl3d_eta' : [],
                'cl3d_phi' : [],
                'cl3d_showerlength' : [],
                'cl3d_coreshowerlength' : [],
                'cl3d_firstlayer' : [],
                'cl3d_seetot' : [],
                'cl3d_seemax' : [],
                'cl3d_spptot' : [],
                'cl3d_sppmax' : [],
                'cl3d_szz' : [],
                'cl3d_srrtot' : [],
                'cl3d_srrmax' : [],
                'cl3d_srrmean' : [],
                'cl3d_hoe' : [],
                'cl3d_meanz' : []
               }

    for evt in range(0, nEntries):
        if evt%100==0: print('--> ',evt)
        # if evt == 500: break

        entry = inChain.GetEntry(evt)
        
        _cl3d_pt               = inChain.cl3d_pt
        _cl3d_energy           = inChain.cl3d_energy
        _cl3d_eta              = inChain.cl3d_eta
        _cl3d_phi              = inChain.cl3d_phi
        _cl3d_showerlength     = inChain.cl3d_showerlength
        _cl3d_coreshowerlength = inChain.cl3d_coreshowerlength
        _cl3d_firstlayer       = inChain.cl3d_firstlayer
        _cl3d_seetot           = inChain.cl3d_seetot
        _cl3d_seemax           = inChain.cl3d_seemax
        _cl3d_spptot           = inChain.cl3d_spptot
        _cl3d_sppmax           = inChain.cl3d_sppmax
        _cl3d_szz              = inChain.cl3d_szz
        _cl3d_srrtot           = inChain.cl3d_srrtot
        _cl3d_srrmax           = inChain.cl3d_srrmax
        _cl3d_srrmean          = inChain.cl3d_srrmean
        _cl3d_hoe              = inChain.cl3d_hoe
        _cl3d_meanz            = inChain.cl3d_meanz

        # loop over all the CLTWs
        for cl3dIdx in range(len(_cl3d_pt)):
            featDict['cl3d_pt'].append( inputQuantizer(_cl3d_pt[cl3dIdx], 0.25, 14) )
            featDict['cl3d_energy'].append( inputQuantizer(_cl3d_energy[cl3dIdx], 0.25, 14) )
            featDict['cl3d_eta'].append( inputQuantizer(abs(_cl3d_eta[cl3dIdx])-1.321, 0.004, 9) ) # transform to local variables (computed form JB schematics -ln(tan(arcos(4565.6/sqrt(4565.6^2+2624.6^2))/2))=1.32070)
            featDict['cl3d_phi'].append( inputQuantizer(_cl3d_phi[cl3dIdx], 0.004, 9) )
            featDict['cl3d_showerlength'].append( _cl3d_showerlength[cl3dIdx] )
            featDict['cl3d_coreshowerlength'].append( _cl3d_coreshowerlength[cl3dIdx] )
            featDict['cl3d_firstlayer'].append( _cl3d_firstlayer[cl3dIdx] )
            featDict['cl3d_seetot'].append( inputQuantizer( _cl3d_seetot[cl3dIdx], 0.0000153, 16) ) # 1/2**16 = 0.0000152587
            featDict['cl3d_seemax'].append( inputQuantizer( _cl3d_seemax[cl3dIdx], 0.0000153, 16) )
            featDict['cl3d_spptot'].append( inputQuantizer( _cl3d_spptot[cl3dIdx], 0.0000153, 16) )
            featDict['cl3d_sppmax'].append( inputQuantizer( _cl3d_sppmax[cl3dIdx], 0.0000153, 16) )
            featDict['cl3d_szz'].append( inputQuantizer( _cl3d_szz[cl3dIdx], 0.00153, 16) ) # 100/2**16 = 0.00152587
            featDict['cl3d_srrtot'].append( inputQuantizer( _cl3d_srrtot[cl3dIdx], 0.0000153, 16) )
            featDict['cl3d_srrmax'].append( inputQuantizer( _cl3d_srrmax[cl3dIdx], 0.0000153, 16) )
            featDict['cl3d_srrmean'].append( inputQuantizer( _cl3d_srrmean[cl3dIdx], 0.0000153, 16) )
            featDict['cl3d_hoe'].append( inputQuantizer( _cl3d_hoe[cl3dIdx], 0.00153, 16) )
            featDict['cl3d_meanz'].append( inputQuantizer(10*(abs(_cl3d_meanz[cl3dIdx])-321.05), 0.5, 12) ) # transform to local variables (based on JB schematics)
            
    allFeats = ['cl3d_pt', 'cl3d_energy', 'cl3d_eta', 'cl3d_phi',
                'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer',
                'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz',
                'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']

    scaler = StandardScaler()

    scalingDF = pd.DataFrame(featDict, columns=allFeats)
    scaledDF = pd.DataFrame(scaler.fit_transform(scalingDF), columns=allFeats)

    save_obj(scaler, options.pathOut+'/cl3d_features_scaler.pkl')

    with open(options.pathOut+'/cl3d_features_scaler.txt', 'w') as f:
        f.write("## feature \t mean \t std ##\n")
        for i, item in enumerate(allFeats):
            f.write(item+" - "+str(scaler.mean_[i])+" - "+str(scaler.scale_[i])+"\n")

