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
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    CLTWimages_CB = []
    CLTWpositions_CB = []
    EMUidscore_CB = []
    EMUcalibpt_CB = []

    CLTWimages_CE = []
    CLTWpositions_CE = []
    CL3Dfeatures_CE = []
    EMUidscore_CE = []
    EMUcalibpt_CE = []

    # loop over the events to fill all the tesors
    print('creating chain')
    inChain = ROOT.TChain('L1CaloTauNtuplizer/L1TauMinatorTree')
    inChain.Add(options.fin)

    jobIdx = options.fin.split('Ntuple_')[1].split('.')[0]

    nEntries = inChain.GetEntries()
    if nEntries == 0: exit()

    for evt in range(0, nEntries):
        if evt%100==0: print('--> ',evt)
        # if evt == 500: break

        entry = inChain.GetEntry(evt)

        _EventNumber = inChain.EventNumber
        
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

        _clNxM_CB_seedEta      = inChain.clNxM_CB_seedEta
        _clNxM_CB_seedPhi      = inChain.clNxM_CB_seedPhi
        _clNxM_CB_towerEm      = inChain.clNxM_CB_towerEm
        _clNxM_CB_towerHad     = inChain.clNxM_CB_towerHad
        _clNxM_CB_towerEgEt    = inChain.clNxM_CB_towerEgEt
        _clNxM_CB_calibPt      = inChain.clNxM_CB_calibPt
        _clNxM_CB_IDscore      = inChain.clNxM_CB_IDscore

        _clNxM_CE_seedEta      = inChain.clNxM_CE_seedEta
        _clNxM_CE_seedPhi      = inChain.clNxM_CE_seedPhi
        _clNxM_CE_towerEm      = inChain.clNxM_CE_towerEm
        _clNxM_CE_towerHad     = inChain.clNxM_CE_towerHad
        _clNxM_CE_towerEgEt    = inChain.clNxM_CE_towerEgEt
        _clNxM_CE_calibPt      = inChain.clNxM_CE_calibPt
        _clNxM_CE_IDscore      = inChain.clNxM_CE_IDscore


        if max(_clNxM_CB_calibPt)>1500:
            print("** ERROR HERE, _clNxM_CB_calibPt", max(_clNxM_CB_calibPt))
        
        if max(_clNxM_CE_calibPt)>1500:
            print("** ERROR HERE, _clNxM_CE_calibPt", max(_clNxM_CE_calibPt))
        
        # print('min _clNxM_CB_calibPt', min(_clNxM_CB_calibPt), 'max _clNxM_CB_calibPt', max(_clNxM_CB_calibPt))
        # print('min _clNxM_CE_calibPt', min(_clNxM_CE_calibPt), 'max _clNxM_CE_calibPt', max(_clNxM_CE_calibPt))
        # print('min _clNxM_CB_IDscore', min(_clNxM_CB_IDscore), 'max _clNxM_CB_IDscore', max(_clNxM_CB_IDscore))
        # print('min _clNxM_CE_IDscore', min(_clNxM_CE_IDscore), 'max _clNxM_CE_IDscore', max(_clNxM_CE_IDscore))



        for idx in range(len(_clNxM_CB_IDscore)):
            clNxM_CB_seedEta   = _clNxM_CB_seedEta[idx]
            clNxM_CB_seedPhi   = _clNxM_CB_seedPhi[idx]
            clNxM_CB_towerEm   = _clNxM_CB_towerEm[idx]
            clNxM_CB_towerHad  = _clNxM_CB_towerHad[idx]
            clNxM_CB_towerEgEt = _clNxM_CB_towerEgEt[idx]
            clNxM_CB_calibPt   = _clNxM_CB_calibPt[idx]
            clNxM_CB_IDscore   = _clNxM_CB_IDscore[idx]

            # CLTW image
            x1l = []
            for j in range(45):
                x1l.append(inputQuantizer_vctd(clNxM_CB_towerEgEt[j], 0.25, 10))
                x1l.append(inputQuantizer_vctd(clNxM_CB_towerEm[j],   0.25, 10))
                x1l.append(inputQuantizer_vctd(clNxM_CB_towerHad[j],  0.25, 10))
            x1 = np.array(x1l).reshape(N,M,3)

            # CLTW position
            x2l = []
            x2l.append(clNxM_CB_seedEta)
            x2l.append(clNxM_CB_seedPhi)
            x2 = np.array(x2l)

            CLTWimages_CB.append(x1)
            CLTWpositions_CB.append(x2)
            EMUidscore_CB.append(clNxM_CB_IDscore)
            EMUcalibpt_CB.append(clNxM_CB_calibPt)

        for idx in range(len(_clNxM_CE_IDscore)):
            clNxM_CE_seedEta   = _clNxM_CE_seedEta[idx]
            clNxM_CE_seedPhi   = _clNxM_CE_seedPhi[idx]
            clNxM_CE_towerEm   = _clNxM_CE_towerEm[idx]
            clNxM_CE_towerHad  = _clNxM_CE_towerHad[idx]
            clNxM_CE_towerEgEt = _clNxM_CE_towerEgEt[idx]
            clNxM_CE_calibPt   = _clNxM_CE_calibPt[idx]
            clNxM_CE_IDscore   = _clNxM_CE_IDscore[idx]

            cl3d_pt               = _cl3d_pt[idx]
            cl3d_energy           = _cl3d_energy[idx]
            cl3d_eta              = _cl3d_eta[idx]
            cl3d_phi              = _cl3d_phi[idx]
            cl3d_showerlength     = _cl3d_showerlength[idx]
            cl3d_coreshowerlength = _cl3d_coreshowerlength[idx]
            cl3d_firstlayer       = _cl3d_firstlayer[idx]
            cl3d_seetot           = _cl3d_seetot[idx]
            cl3d_seemax           = _cl3d_seemax[idx]
            cl3d_spptot           = _cl3d_spptot[idx]
            cl3d_sppmax           = _cl3d_sppmax[idx]
            cl3d_szz              = _cl3d_szz[idx]
            cl3d_srrtot           = _cl3d_srrtot[idx]
            cl3d_srrmax           = _cl3d_srrmax[idx]
            cl3d_srrmean          = _cl3d_srrmean[idx]
            cl3d_hoe              = _cl3d_hoe[idx]
            cl3d_meanz            = _cl3d_meanz[idx]
            cl3d_quality          = _cl3d_quality[idx]

            # CLTW image
            x1l = []
            for j in range(45):
                x1l.append(inputQuantizer_vctd(clNxM_CE_towerEgEt[j], 0.25, 10))
                x1l.append(inputQuantizer_vctd(clNxM_CE_towerEm[j],   0.25, 10))
                x1l.append(inputQuantizer_vctd(clNxM_CE_towerHad[j],  0.25, 10))
            x1 = np.array(x1l).reshape(N,M,3)

            # CLTW position
            x2l = []
            x2l.append(clNxM_CE_seedEta)
            x2l.append(clNxM_CE_seedPhi)
            x2 = np.array(x2l)

            # CL3D features
            x3 = np.array([(inputQuantizer(cl3d_pt,     0.25, 14)                - 6.07127) / 8.09958,
                           # inputQuantizer(cl3d_energy, 0.25, 14),
                           (inputQuantizer(abs(cl3d_eta)-1.321, 0.004, 9)        - 1.43884) / 0.31367,
                           # inputQuantizer(cl3d_phi, 0.004, 9),
                           (cl3d_showerlength                                    - 31.2058) / 7.66842,
                           (cl3d_coreshowerlength                                - 10.0995) / 2.73062,
                           # cl3d_firstlayer,
                           # inputQuantizer( cl3d_seetot, 0.0000153, 16),
                           # inputQuantizer( cl3d_seemax, 0.0000153, 16),
                           (inputQuantizer( cl3d_spptot, 0.0000153, 16)          - 0.02386) / 0.01520,
                           # inputQuantizer( cl3d_sppmax, 0.0000153, 16),
                           (inputQuantizer( cl3d_szz, 0.00153, 16)               - 19.5851) / 12.7077,
                           (inputQuantizer( cl3d_srrtot, 0.0000153, 16)          - 0.00606) / 0.00129,
                           # inputQuantizer( cl3d_srrmax, 0.0000153, 16),
                           # inputQuantizer( cl3d_srrmean, 0.0000153, 16),
                           # inputQuantizer( cl3d_hoe, 0.00153, 16),
                           (inputQuantizer(10*(abs(cl3d_meanz)-321.05), 0.5, 12) - 215.552) / 104.794
                          ])

            CLTWimages_CE.append(x1)
            CLTWpositions_CE.append(x2)
            CL3Dfeatures_CE.append(x3)
            EMUidscore_CE.append(clNxM_CE_IDscore)
            EMUcalibpt_CE.append(clNxM_CE_calibPt)


    # end of the loop over the events
    #################################

    CLTWimages_CB = np.array(CLTWimages_CB)
    CLTWpositions_CB = np.array(CLTWpositions_CB)
    EMUidscore_CB = np.array(EMUidscore_CB)
    EMUcalibpt_CB = np.array(EMUcalibpt_CB)

    CLTWimages_CE = np.array(CLTWimages_CE)
    CLTWpositions_CE = np.array(CLTWpositions_CE)
    CL3Dfeatures_CE = np.array(CL3Dfeatures_CE)
    EMUidscore_CE = np.array(EMUidscore_CE)
    EMUcalibpt_CE = np.array(EMUcalibpt_CE)

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
                   'EMUidscore_CB': outdir+'/barrel/EMUidscore_'+options.caloClNxM+'_CB_'+jobIdx+'.npz',
                   'EMUcalibpt_CB': outdir+'/barrel/EMUcalibpt_'+options.caloClNxM+'_CB_'+jobIdx+'.npz',

                   'CLTWimages_CE': outdir+'/endcap/CLTWimages_'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'CLTWpositions_CE': outdir+'/endcap/CLTWpositions_'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'CL3Dfeatures_CE': outdir+'/endcap/CL3Dfeatures_cltw'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'EMUidscore_CE': outdir+'/endcap/EMUidscore_'+options.caloClNxM+'_CE_'+jobIdx+'.npz',
                   'EMUcalibpt_CE': outdir+'/endcap/EMUcalibpt_'+options.caloClNxM+'_CE_'+jobIdx+'.npz'
                 }

    np.savez_compressed(saveTensTo['CLTWimages_CB'], CLTWimages_CB)
    np.savez_compressed(saveTensTo['CLTWpositions_CB'], CLTWpositions_CB)
    np.savez_compressed(saveTensTo['EMUidscore_CB'], EMUidscore_CB)
    np.savez_compressed(saveTensTo['EMUcalibpt_CB'], EMUcalibpt_CB)

    np.savez_compressed(saveTensTo['CLTWimages_CE'], CLTWimages_CE)
    np.savez_compressed(saveTensTo['CLTWpositions_CE'], CLTWpositions_CE)
    np.savez_compressed(saveTensTo['CL3Dfeatures_CE'], CL3Dfeatures_CE)
    np.savez_compressed(saveTensTo['EMUidscore_CE'], EMUidscore_CE)
    np.savez_compressed(saveTensTo['EMUcalibpt_CE'], EMUcalibpt_CE)
