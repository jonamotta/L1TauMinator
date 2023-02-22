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
    parser.add_option("--NtupleV",          dest="NtupleV",                default=None)
    parser.add_option("--v",                dest="v",                      default=None)
    parser.add_option("--date",             dest="date",                   default=None)
    parser.add_option('--caloClNxM',        dest='caloClNxM',              default="5x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    CLTWimages_barrel_ID = []
    CLTWpositions_barrel_ID = []
    Y_barrel_ID = []

    CLTWimages_barrel_CAL = []
    CLTWpositions_barrel_CAL = []
    Y_barrel_CAL = []

    CLTWimages_endcap_ID = []
    CLTWpositions_endcap_ID = []
    CL3Dfeatures_endcap_ID = []
    Y_endcap_ID = []

    CLTWimages_endcap_CAL = []
    CLTWpositions_endcap_CAL = []
    CL3Dfeatures_endcap_CAL = []
    Y_endcap_CAL = []

    # loop over the events to fill all the tesors
    print('creating chain')
    directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT/'
    inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
    inChain.Add(directory+'/Ntuple_*9*.root');
    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%100==0: print('--> ',evt)
        if evt == 2000: break

        entry = inChain.GetEntry(evt)

        _EventNumber = inChain.EventNumber

        _gentau_visEta = inChain.tau_visEta
        if len(_gentau_visEta)==0: continue
        _gentau_visPhi = inChain.tau_visPhi
        _gentau_visPt = inChain.tau_visPt
        _gentau_dm = inChain.tau_DM
        
        _cl3d_pt = inChain.cl3d_pt
        _cl3d_energy = inChain.cl3d_energy
        _cl3d_eta = inChain.cl3d_eta
        _cl3d_phi = inChain.cl3d_phi
        _cl3d_showerlength = inChain.cl3d_showerlength
        _cl3d_coreshowerlength = inChain.cl3d_coreshowerlength
        _cl3d_firstlayer = inChain.cl3d_firstlayer
        _cl3d_seetot = inChain.cl3d_seetot
        _cl3d_seemax = inChain.cl3d_seemax
        _cl3d_spptot = inChain.cl3d_spptot
        _cl3d_sppmax = inChain.cl3d_sppmax
        _cl3d_szz = inChain.cl3d_szz
        _cl3d_srrtot = inChain.cl3d_srrtot
        _cl3d_srrmax = inChain.cl3d_srrmax
        _cl3d_srrmean = inChain.cl3d_srrmean
        _cl3d_hoe = inChain.cl3d_hoe
        _cl3d_meanz = inChain.cl3d_meanz
        _cl3d_quality = inChain.cl3d_quality
        _cl3d_stale = np.repeat(False, len(list(_cl3d_quality)))
        
        _cl5x9_seedEta = inChain.cl5x9_seedEta
        _cl5x9_seedPhi = inChain.cl5x9_seedPhi
        _cl5x9_towerEm = inChain.cl5x9_towerEm
        _cl5x9_towerHad = inChain.cl5x9_towerHad
        _cl5x9_towerEgEt = inChain.cl5x9_towerEgEt


        # loop over all the CLTWs
        for cl5x9_seedEta, cl5x9_seedPhi, cl5x9_towerEm, cl5x9_towerHad, cl5x9_towerEgEt in zip(_cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_towerEm, _cl5x9_towerHad, _cl5x9_towerEgEt):
            # if in the encap loop over the CL3Ds to create the composite objects
            if abs(cl5x9_seedEta) > 1.5:
                highestPt = -99.9
                cl3d_matches_cl5x9 = False
                cl3dIdx = 0
                matched_cl3dIdx = -99
                for cl3d_pt, cl3d_energy, cl3d_eta, cl3d_phi, cl3d_showerlength, cl3d_coreshowerlength, cl3d_firstlayer, cl3d_seetot, cl3d_seemax, cl3d_spptot, cl3d_sppmax, cl3d_szz, cl3d_srrtot, cl3d_srrmax, cl3d_srrmean, cl3d_hoe, cl3d_meanz, cl3d_quality in zip(_cl3d_pt, _cl3d_energy, _cl3d_eta, _cl3d_phi, _cl3d_showerlength, _cl3d_coreshowerlength, _cl3d_firstlayer, _cl3d_seetot, _cl3d_seemax, _cl3d_spptot, _cl3d_sppmax, _cl3d_szz, _cl3d_srrtot, _cl3d_srrmax, _cl3d_srrmean, _cl3d_hoe, _cl3d_meanz, _cl3d_quality):
                    if _cl3d_stale[cl3dIdx]: continue # skip CL3Ds that have already been used

                    dR2 = (cl5x9_seedEta-cl3d_eta)**2 + deltaPhi(cl5x9_seedPhi, cl3d_phi)**2
                    if dR2 <= 0.25 and cl3d_pt > highestPt:
                        highestPt = cl3d_pt
                        cl3d_matches_cl5x9 = True
                        matched_cl3dIdx = cl3dIdx
                        localAbsEta = abs(cl3d_eta)-1.45
                        localAbsMeanZ = 10*(abs(cl3d_meanz)-320)
                        cl3d_feats = [cl3d_pt, cl3d_energy, localAbsEta, cl3d_phi, cl3d_showerlength, cl3d_coreshowerlength, cl3d_firstlayer, cl3d_seetot, cl3d_seemax, cl3d_spptot, cl3d_sppmax, cl3d_szz, cl3d_srrtot, cl3d_srrmax, cl3d_srrmean, cl3d_hoe, localAbsMeanZ, cl3d_quality]

                    cl3dIdx += 1

                # mark used CL3Ds as stale to avoid overlap
                if matched_cl3dIdx != -99: _cl3d_stale[matched_cl3dIdx] = True

            matched = False
            for tauPt, tauEta, tauPhi, tauDM in zip(_gentau_visPt, _gentau_visEta, _gentau_visPhi, _gentau_dm):
                if abs(tauEta)>3.0: continue
                if tauPt < 18: continue

                dR2min = 0.25
                dR2 = (cl5x9_seedEta-tauEta)**2 + deltaPhi(cl5x9_seedPhi, tauPhi)**2
                if dR2 <= dR2min:
                    dR2min = dR2
                    matched = True
                    tau_feats = [tauPt, tauEta, tauPhi, tauDM]

            # fill tensors with the signal
            if matched:
                # fill tensors for the endcap
                if abs(cl5x9_seedEta) > 1.5:
                    # CLTW image
                    x1l = []
                    for j in range(45):
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEgEt[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEm[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerHad[j], 0.25, 10))
                    x1 = np.array(x1l).reshape(N,M,3)

                    # CLTW position
                    x2l = []
                    x2l.append(cl5x9_seedEta)
                    x2l.append(cl5x9_seedPhi)
                    x2 = np.array(x2l)

                    # CL3D features
                    x3 = np.array(cl3d_feats)

                    # tau target features
                    yl = tau_feats
                    yl.append(1)
                    y = np.array(yl)

                    CLTWimages_endcap_ID.append(x1)
                    CLTWpositions_endcap_ID.append(x2)
                    CL3Dfeatures_endcap_ID.append(x3)
                    Y_endcap_ID.append(y)

                    CLTWimages_endcap_CAL.append(x1)
                    CLTWpositions_endcap_CAL.append(x2)
                    CL3Dfeatures_endcap_CAL.append(x3)
                    Y_endcap_CAL.append(y)

                # fill tensors for the barrel
                elif matched:
                    # CLTW image
                    x1l = []
                    for j in range(45):
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEgEt[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEm[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerHad[j], 0.25, 10))
                    x1 = np.array(x1l).reshape(N,M,3)

                    # CLTW position
                    x2l = []
                    x2l.append(cl5x9_seedEta)
                    x2l.append(cl5x9_seedPhi)
                    x2 = np.array(x2l)

                    # tau target features
                    yl = tau_feats
                    yl.append(1)
                    y = np.array(yl)

                    CLTWimages_barrel_ID.append(x1)
                    CLTWpositions_barrel_ID.append(x2)
                    Y_barrel_ID.append(y)

                    CLTWimages_barrel_CAL.append(x1)
                    CLTWpositions_barrel_CAL.append(x2)
                    Y_barrel_CAL.append(y)

            # fill tensors for endcap/barrel with the PU
            else:
                if abs(cl5x9_seedEta) > 1.5:
                    # CLTW image
                    x1l = []
                    for j in range(45):
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEgEt[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEm[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerHad[j], 0.25, 10))
                    x1 = np.array(x1l).reshape(N,M,3)

                    # CLTW position
                    x2l = []
                    x2l.append(cl5x9_seedEta)
                    x2l.append(cl5x9_seedPhi)
                    x2 = np.array(x2l)

                    # CL3D features
                    x3 = np.array(cl3d_feats)

                    # tau target features
                    yl = [-99., -99., -99., -99., 0]
                    y = np.array(yl)

                    CLTWimages_endcap_ID.append(x1)
                    CLTWpositions_endcap_ID.append(x2)
                    CL3Dfeatures_endcap_ID.append(x3)
                    Y_endcap_ID.append(y)

                else:
                    # CLTW image
                    x1l = []
                    for j in range(45):
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEgEt[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerEm[j], 0.25, 10))
                        x1l.append(inputQuantizer_vctd(cl5x9_towerHad[j], 0.25, 10))
                    x1 = np.array(x1l).reshape(N,M,3)

                    # CLTW position
                    x2l = []
                    x2l.append(cl5x9_seedEta)
                    x2l.append(cl5x9_seedPhi)
                    x2 = np.array(x2l)

                    # tau target features
                    yl = [-99., -99., -99., -99., 0]
                    y = np.array(yl)

                    CLTWimages_barrel_ID.append(x1)
                    CLTWpositions_barrel_ID.append(x2)
                    Y_barrel_ID.append(y)

    # end of the loop over the events
    #################################

    CLTWimages_barrel_ID_tens = np.array(CLTWimages_barrel_ID)
    CLTWpositions_barrel_ID_tens = np.array(CLTWpositions_barrel_ID)
    Y_barrel_ID_tens = np.array(Y_barrel_ID)

    CLTWimages_barrel_CAL_tens = np.array(CLTWimages_barrel_CAL)
    CLTWpositions_barrel_CAL_tens = np.array(CLTWpositions_barrel_CAL)
    Y_barrel_CAL_tens = np.array(Y_barrel_CAL)

    CLTWimages_endcap_ID_tens = np.array(CLTWimages_endcap_ID)
    CLTWpositions_endcap_ID_tens = np.array(CLTWpositions_endcap_ID)
    CL3Dfeatures_endcap_ID_tens = np.array(CL3Dfeatures_endcap_ID)
    Y_endcap_ID_tens = np.array(Y_endcap_ID)

    CLTWimages_endcap_CAL_tens = np.array(CLTWimages_endcap_CAL)
    CLTWpositions_endcap_CAL_tens = np.array(CLTWpositions_endcap_CAL)
    CL3Dfeatures_endcap_CAL_tens = np.array(CL3Dfeatures_endcap_CAL)
    Y_endcap_CAL_tens = np.array(Y_endcap_CAL)

    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/'
    os.system('mkdir -p '+outdir+'/barrel')
    os.system('mkdir -p '+outdir+'/endcap')
    saveTensTo = { 'CLTWimages_barrel_ID': outdir+'/barrel/CLTWimages_barrel_ID.npz',
                   'CLTWpositions_barrel_ID': outdir+'/barrel/CLTWpositions_barrel_ID.npz',
                   'Y_barrel_ID': outdir+'/barrel/Y_barrel_ID.npz',
                   'CLTWimages_barrel_CAL': outdir+'/barrel/CLTWimages_barrel_CAL.npz',
                   'CLTWpositions_barrel_CAL': outdir+'/barrel/CLTWpositions_barrel_CAL.npz',
                   'Y_barrel_CAL': outdir+'/barrel/Y_barrel_CAL.npz',
                   'CLTWimages_endcap_ID': outdir+'/endcap/CLTWimages_endcap_ID.npz',
                   'CLTWpositions_endcap_ID': outdir+'/endcap/CLTWpositions_endcap_ID.npz',
                   'CL3Dfeatures_endcap_ID': outdir+'/endcap/CL3Dfeatures_endcap_ID.npz',
                   'Y_endcap_ID': outdir+'/endcap/Y_endcap_ID.npz',
                   'CLTWimages_endcap_CAL': outdir+'/endcap/CLTWimages_endcap_CAL.npz',
                   'CLTWpositions_endcap_CAL': outdir+'/endcap/CLTWpositions_endcap_CAL.npz',
                   'CL3Dfeatures_endcap_CAL': outdir+'/endcap/CL3Dfeatures_endcap_CAL.npz',
                   'Y_endcap_CAL': outdir+'/endcap/Y_endcap_CAL.npz'}

    np.savez_compressed(saveTensTo['CLTWimages_barrel_ID'], CLTWimages_barrel_ID_tens)
    np.savez_compressed(saveTensTo['CLTWpositions_barrel_ID'], CLTWpositions_barrel_ID_tens)
    np.savez_compressed(saveTensTo['Y_barrel_ID'], Y_barrel_ID_tens)

    np.savez_compressed(saveTensTo['CLTWimages_barrel_CAL'], CLTWimages_barrel_CAL_tens)
    np.savez_compressed(saveTensTo['CLTWpositions_barrel_CAL'], CLTWpositions_barrel_CAL_tens)
    np.savez_compressed(saveTensTo['Y_barrel_CAL'], Y_barrel_CAL_tens)

    np.savez_compressed(saveTensTo['CLTWimages_endcap_ID'], CLTWimages_endcap_ID_tens)
    np.savez_compressed(saveTensTo['CLTWpositions_endcap_ID'], CLTWpositions_endcap_ID_tens)
    np.savez_compressed(saveTensTo['CL3Dfeatures_endcap_ID'], CL3Dfeatures_endcap_ID_tens)
    np.savez_compressed(saveTensTo['Y_endcap_ID'], Y_endcap_ID_tens)

    np.savez_compressed(saveTensTo['CLTWimages_endcap_CAL'], CLTWimages_endcap_CAL_tens)
    np.savez_compressed(saveTensTo['CLTWpositions_endcap_CAL'], CLTWpositions_endcap_CAL_tens)
    np.savez_compressed(saveTensTo['CL3Dfeatures_endcap_CAL'], CL3Dfeatures_endcap_CAL_tens)
    np.savez_compressed(saveTensTo['Y_endcap_CAL'], Y_endcap_CAL_tens)



   