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
    parser.add_option('--fin',       dest='fin',       default=None)
    parser.add_option("--fout",      dest="fout",      default=None)
    parser.add_option('--caloClNxM', dest='caloClNxM', default="5x9")
    parser.add_option("--uTauPtCut", dest="uTauPtCut", default=None, type=int)
    parser.add_option("--lTauPtCut", dest="lTauPtCut", default=None, type=int)
    parser.add_option("--CBCEsplit", dest="CBCEsplit", default=1.5, type=float)
    parser.add_option("--uEtacut",   dest="uEtacut",   default=None, type=float)
    parser.add_option("--lEtacut",   dest="lEtacut",   default=None, type=float)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    EtaPhiLsb = np.round(np.pi/36, 5)

    CLTWimages_CB = []
    CLTWpositions_CB = []
    Y_CB = []

    CLTWimages_CE = []
    CLTWpositions_CE = []
    CL3Dfeatures_CE = []
    Y_CE = []

    CLTWphiFlippedFlag_CB = []
    CLTWphiFlippedFlag_CE = []

    # loop over the events to fill all the tesors
    print('creating chain')
    inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
    inChain.Add(options.fin)

    jobIdx = options.fin.split('Ntuple_')[1].split('.')[0]

    nEntries = inChain.GetEntries()
    if nEntries == 0: exit()

    for evt in range(0, nEntries):
        if evt%100==0: print('--> ',evt)
        # if evt == 500: break

        entry = inChain.GetEntry(evt)

        _EventNumber = inChain.EventNumber

        _gentau_visEta = inChain.tau_visEta
        if len(_gentau_visEta)==0: continue
        _gentau_visPhi = inChain.tau_visPhi
        _gentau_visPt = inChain.tau_visPt
        _gentau_dm = inChain.tau_DM
        
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
        _cl3d_quality          = inChain.cl3d_quality

        _cl5x9_seedEta      = inChain.cl5x9_seedEta
        _cl5x9_seedPhi      = inChain.cl5x9_seedPhi
        _cl5x9_towerEm      = inChain.cl5x9_towerEm
        _cl5x9_towerHad     = inChain.cl5x9_towerHad
        _cl5x9_towerEgEt    = inChain.cl5x9_towerEgEt
        _cl5x9_isPhiFlipped = inChain.cl5x9_isPhiFlipped
        _cl5x9_cl3dMatchIdx = inChain.cl5x9_cl3dMatchIdx
        _cl5x9_tauMatchIdx  = inChain.cl5x9_tauMatchIdx

        # loop over all the CLTWs
        for cl5x9_seedEta, cl5x9_seedPhi, cl5x9_towerEm, cl5x9_towerHad, cl5x9_towerEgEt, isPhiFlipped, cl3dMatchIdx, tauMatchIdx in zip(_cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_towerEm, _cl5x9_towerHad, _cl5x9_towerEgEt, _cl5x9_isPhiFlipped, _cl5x9_cl3dMatchIdx, _cl5x9_tauMatchIdx):
            
            # skip the phiFlipped clusters if they are bkg
            if isPhiFlipped and tauMatchIdx < 0: continue

            # apply minimum pt cut on taus
            if options.lTauPtCut and tauMatchIdx >= 0 and _gentau_visPt[tauMatchIdx] < options.lTauPtCut: continue

            # apply maximum eta cut on taus and on l1 objects
            
            if options.uEtacut and abs(cl5x9_seedEta) > options.uEtacut: continue

            # if in the barrel just save the info of the CLTW
            if abs(cl5x9_seedEta) < options.CBCEsplit:
                # CLTW image
                x1l = []
                for j in range(45):
                    x1l.append(inputQuantizer_vctd(cl5x9_towerEgEt[j], 0.25, 10))
                    x1l.append(inputQuantizer_vctd(cl5x9_towerEm[j],   0.25, 10))
                    x1l.append(inputQuantizer_vctd(cl5x9_towerHad[j],  0.25, 10))
                x1 = np.array(x1l).reshape(N,M,3)

                # CLTW position
                x2l = []
                x2l.append(inputQuantizer(cl5x9_seedEta, EtaPhiLsb, 7))
                x2l.append(inputQuantizer(cl5x9_seedPhi, EtaPhiLsb, 7))
                x2 = np.array(x2l)

                # tau target features
                yl = [-99., 0, -99., -99., -99.]
                if tauMatchIdx != -99:
                    yl = [_gentau_visPt[tauMatchIdx],
                          1,
                          _gentau_visEta[tauMatchIdx],
                          _gentau_visPhi[tauMatchIdx],
                          _gentau_dm[tauMatchIdx]
                         ]
                y = np.array(yl)

                CLTWimages_CB.append(x1)
                CLTWpositions_CB.append(x2)
                Y_CB.append(y)

                CLTWphiFlippedFlag_CB.append(isPhiFlipped)

            # if in the endcap save the info of the CLTW and of the CL3D
            else:
                # skip CLTW that are not matched to a CL3D
                if cl3dMatchIdx < 0: continue

                # CLTW image
                x1l = []
                for j in range(45):
                    x1l.append(inputQuantizer_vctd(cl5x9_towerEgEt[j], 0.25, 10))
                    x1l.append(inputQuantizer_vctd(cl5x9_towerEm[j],   0.25, 10))
                    x1l.append(inputQuantizer_vctd(cl5x9_towerHad[j],  0.25, 10))
                x1 = np.array(x1l).reshape(N,M,3)

                # CLTW position
                x2l = []
                x2l.append(inputQuantizer(cl5x9_seedEta, EtaPhiLsb, 7))
                x2l.append(inputQuantizer(cl5x9_seedPhi, EtaPhiLsb, 7))
                x2 = np.array(x2l)

                # CL3D features
                x3 = np.array([inputQuantizer(_cl3d_pt[cl3dMatchIdx],     0.25, 14),
                               inputQuantizer(_cl3d_energy[cl3dMatchIdx], 0.25, 14),
                               inputQuantizer(abs(_cl3d_eta[cl3dMatchIdx])-1.321, 0.004, 9), # transform to local variables (computed form JB schematics -ln(tan(arcos(4565.6/sqrt(4565.6^2+2624.6^2))/2))=1.32070)
                               inputQuantizer(_cl3d_phi[cl3dMatchIdx], 0.004, 9),
                               _cl3d_showerlength[cl3dMatchIdx],
                               _cl3d_coreshowerlength[cl3dMatchIdx],
                               _cl3d_firstlayer[cl3dMatchIdx],
                               inputQuantizer( _cl3d_seetot[cl3dMatchIdx], 0.0000153, 16), # 1/2**16 = 0.0000152587
                               inputQuantizer( _cl3d_seemax[cl3dMatchIdx], 0.0000153, 16),
                               inputQuantizer( _cl3d_spptot[cl3dMatchIdx], 0.0000153, 16),
                               inputQuantizer( _cl3d_sppmax[cl3dMatchIdx], 0.0000153, 16),
                               inputQuantizer( _cl3d_szz[cl3dMatchIdx], 0.00153, 16), # 100/2**16 = 0.00152587
                               inputQuantizer( _cl3d_srrtot[cl3dMatchIdx], 0.0000153, 16),
                               inputQuantizer( _cl3d_srrmax[cl3dMatchIdx], 0.0000153, 16),
                               inputQuantizer( _cl3d_srrmean[cl3dMatchIdx], 0.0000153, 16),
                               inputQuantizer( _cl3d_hoe[cl3dMatchIdx], 0.00153, 16),
                               inputQuantizer(10*(abs(_cl3d_meanz[cl3dMatchIdx])-321.05), 0.5, 12) # transform to local variables (based on JB schematics)
                              ])

                # tau target features
                yl = [-99., 0, -99., -99., -99.]
                if tauMatchIdx != -99:
                    yl = [_gentau_visPt[tauMatchIdx],
                          1,
                          _gentau_visEta[tauMatchIdx],
                          _gentau_visPhi[tauMatchIdx],
                          _gentau_dm[tauMatchIdx]
                         ]
                y = np.array(yl)

                CLTWimages_CE.append(x1)
                CLTWpositions_CE.append(x2)
                CL3Dfeatures_CE.append(x3)
                Y_CE.append(y)

                CLTWphiFlippedFlag_CE.append(isPhiFlipped)

    # end of the loop over the events
    #################################

    CLTWimages_CB = np.array(CLTWimages_CB)
    CLTWpositions_CB = np.array(CLTWpositions_CB)
    Y_CB = np.array(Y_CB)

    CLTWimages_CE = np.array(CLTWimages_CE)
    CLTWpositions_CE = np.array(CLTWpositions_CE)
    CL3Dfeatures_CE = np.array(CL3Dfeatures_CE)
    Y_CE = np.array(Y_CE)

    ## DEBUG
    # print(CLTWimages_CB.shape)
    # print(CLTWpositions_CB.shape)
    # print(Y_CB.shape)
    # print('---------------')
    # print(CLTWimages_CE.shape)
    # print(CLTWpositions_CE.shape)
    # print(CL3Dfeatures_CE.shape)
    # print(Y_CE.shape)
    # exit()

    outdir = options.fout
    os.system('mkdir -p '+outdir+'/barrel')
    os.system('mkdir -p '+outdir+'/endcap')
    saveTensTo = { 'CLTWimages_CB': outdir+'/barrel/CLTWimages_'+options.caloClNxM+'_CB_'+jobIdx+'.npz',
                   'CLTWpositions_CB': outdir+'/barrel/CLTWpositions_'+options.caloClNxM+'_CB_'+jobIdx+'.npz',
                   'Y_CB': outdir+'/barrel/Y_cltw'+options.caloClNxM+'_CB_'+jobIdx+'.npz',
                   'FlippedFlag_CB': outdir+'/barrel/FlippedFlag_cltw'+options.caloClNxM+'_CB_'+jobIdx+'.npz',

                   'CLTWimages_CE': outdir+'/endcap/CLTWimages_'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'CLTWpositions_CE': outdir+'/endcap/CLTWpositions_'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'CL3Dfeatures_CE': outdir+'/endcap/CL3Dfeatures_cltw'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'Y_CE': outdir+'/endcap/Y_cltw'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'FlippedFlag_CE': outdir+'/endcap/FlippedFlag_cltw'+options.caloClNxM+'_CE_'+jobIdx+'.npz'
                 }

    np.savez_compressed(saveTensTo['CLTWimages_CB'], CLTWimages_CB)
    np.savez_compressed(saveTensTo['CLTWpositions_CB'], CLTWpositions_CB)
    np.savez_compressed(saveTensTo['Y_CB'], Y_CB)
    np.savez_compressed(saveTensTo['FlippedFlag_CB'], CLTWphiFlippedFlag_CB)

    np.savez_compressed(saveTensTo['CLTWimages_CE'], CLTWimages_CE)
    np.savez_compressed(saveTensTo['CLTWpositions_CE'], CLTWpositions_CE)
    np.savez_compressed(saveTensTo['CL3Dfeatures_CE'], CL3Dfeatures_CE)
    np.savez_compressed(saveTensTo['Y_CE'], Y_CE)
    np.savez_compressed(saveTensTo['FlippedFlag_CE'], CLTWphiFlippedFlag_CE)
