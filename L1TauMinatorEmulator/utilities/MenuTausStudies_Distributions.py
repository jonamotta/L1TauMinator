from optparse import OptionParser
from array import array
import numpy as np
import ROOT
import sys
import os

from matplotlib.cm import get_cmap
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

def deltaPhi( phi1, phi2 ):
    delta_phi = np.abs(phi1-phi2)
    delta_sgn = np.sign(phi1-phi2)
    if delta_phi > np.pi: return delta_sgn * (2*np.pi - delta_phi)
    else:                 return delta_sgn * delta_phi


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
    parser.add_option('--WP',               dest='WP',             default='90')
    parser.add_option('--caloClNxM',        dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",        dest="seedEtCut",      default="2p5")
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)


    gentausPt_highPt = []
    gentausPt_restPt = []

    matched_highPt_CB_seedEta = []
    matched_highPt_CB_seedPhi = []
    matched_highPt_CB_towerEm = []
    matched_highPt_CB_towerHad = []
    matched_highPt_CB_towerEgEt = []
    matched_highPt_CB_calibPt = []
    matched_highPt_CB_IDscore = []
    matched_highPt_CB_totalEt = []

    matched_highPt_CE_seedEta = []
    matched_highPt_CE_seedPhi = []
    matched_highPt_CE_towerEm = []
    matched_highPt_CE_towerHad = []
    matched_highPt_CE_towerEgEt = []
    matched_highPt_CE_calibPt = []
    matched_highPt_CE_IDscore = []
    matched_highPt_CE_totalEt = []

    matched_highPt_cl3d_pt = []
    matched_highPt_cl3d_eta = []
    matched_highPt_cl3d_showerlength = []
    matched_highPt_cl3d_coreshowerlength = []
    matched_highPt_cl3d_spptot = []
    matched_highPt_cl3d_szz = []
    matched_highPt_cl3d_srrtot = []
    matched_highPt_cl3d_meanz = []


    matched_restPt_CB_seedEta = []
    matched_restPt_CB_seedPhi = []
    matched_restPt_CB_towerEm = []
    matched_restPt_CB_towerHad = []
    matched_restPt_CB_towerEgEt = []
    matched_restPt_CB_calibPt = []
    matched_restPt_CB_IDscore = []
    matched_restPt_CB_totalEt = []

    matched_restPt_CE_seedEta = []
    matched_restPt_CE_seedPhi = []
    matched_restPt_CE_towerEm = []
    matched_restPt_CE_towerHad = []
    matched_restPt_CE_towerEgEt = []
    matched_restPt_CE_calibPt = []
    matched_restPt_CE_IDscore = []
    matched_restPt_CE_totalEt = []

    matched_restPt_cl3d_pt = []
    matched_restPt_cl3d_eta = []
    matched_restPt_cl3d_showerlength = []
    matched_restPt_cl3d_coreshowerlength = []
    matched_restPt_cl3d_spptot = []
    matched_restPt_cl3d_szz = []
    matched_restPt_cl3d_srrtot = []
    matched_restPt_cl3d_meanz = []

    version = '3'
    user = os.getcwd().split('/')[5]
    directory = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
    inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
    inChain.Add(directory+'/Ntuple_*.root')

    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%1000==0: print('--> ',evt)
        # if evt == 200: break

        entry = inChain.GetEntry(evt)

        _gentau_visEta = list(inChain.tau_visEta)
        _gentau_visPhi = list(inChain.tau_visPhi)
        _gentau_visPt = list(inChain.tau_visPt)

        if len(_gentau_visEta) == 0: continue

        maxPt = max(_gentau_visPt)
        _gentau_highPt_visEta = [ _gentau_visEta[_gentau_visPt.index(maxPt)] ]
        _gentau_highPt_visPhi = [ _gentau_visPhi[_gentau_visPt.index(maxPt)] ]
        _gentau_highPt_visPt = [ _gentau_visPt[_gentau_visPt.index(maxPt)] ]

        del _gentau_visEta[_gentau_visPt.index(maxPt)]
        del _gentau_visPhi[_gentau_visPt.index(maxPt)]
        del _gentau_visPt[_gentau_visPt.index(maxPt)]
        _gentau_restPt_visEta = _gentau_visEta
        _gentau_restPt_visPhi = _gentau_visPhi
        _gentau_restPt_visPt = _gentau_visPt

        gentausPt_highPt.extend(_gentau_highPt_visPt)
        gentausPt_restPt.extend(_gentau_restPt_visPt)

        _clNxM_CB_seedEta = list(inChain.clNxM_CB_seedEta)
        _clNxM_CB_seedPhi = list(inChain.clNxM_CB_seedPhi)
        _clNxM_CB_towerEm = list(inChain.clNxM_CB_towerEm)
        _clNxM_CB_towerHad = list(inChain.clNxM_CB_towerHad)
        _clNxM_CB_towerEgEt = list(inChain.clNxM_CB_towerEgEt)
        _clNxM_CB_calibPt = list(inChain.clNxM_CB_calibPt)
        _clNxM_CB_IDscore = list(inChain.clNxM_CB_IDscore)

        _clNxM_CE_seedEta = list(inChain.clNxM_CE_seedEta)
        _clNxM_CE_seedPhi = list(inChain.clNxM_CE_seedPhi)
        _clNxM_CE_towerEm = list(inChain.clNxM_CE_towerEm)
        _clNxM_CE_towerHad = list(inChain.clNxM_CE_towerHad)
        _clNxM_CE_towerEgEt = list(inChain.clNxM_CE_towerEgEt)
        _clNxM_CE_calibPt = list(inChain.clNxM_CE_calibPt)
        _clNxM_CE_IDscore = list(inChain.clNxM_CE_IDscore)

        _cl3d_pt = list(inChain.cl3d_pt)
        _cl3d_eta = list(inChain.cl3d_eta)
        _cl3d_showerlength = list(inChain.cl3d_showerlength)
        _cl3d_coreshowerlength = list(inChain.cl3d_coreshowerlength)
        _cl3d_spptot = list(inChain.cl3d_spptot)
        _cl3d_szz = list(inChain.cl3d_szz)
        _cl3d_srrtot = list(inChain.cl3d_srrtot)
        _cl3d_meanz = list(inChain.cl3d_meanz)


        for tauPt, tauEta, tauPhi,  in zip(_gentau_highPt_visPt, _gentau_highPt_visEta, _gentau_highPt_visPhi):

            if tauPt < 50: continue

            gentau = ROOT.TLorentzVector()
            gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

            if abs(gentau.Eta()) > options.etaEr: continue # skip taus out of acceptance

            if abs(gentau.Eta()) < 1.5:
                matchedIdx = -99
                highestL1Pt = -99

                for idx in range(len(_clNxM_CB_seedEta)):
                    
                    clNxM_CB_seedEta = _clNxM_CB_seedEta[idx]
                    clNxM_CB_seedPhi = _clNxM_CB_seedPhi[idx]
                    clNxM_CB_totalEt = sum(_clNxM_CB_towerEm[idx]) + sum(_clNxM_CB_towerHad[idx]) + sum(_clNxM_CB_towerEgEt[idx])

                    dEta = gentau.Eta() - clNxM_CB_seedEta
                    dPhi = deltaPhi(gentau.Phi(), clNxM_CB_seedPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.09 and clNxM_CB_totalEt > highestL1Pt:
                        highestL1Pt = clNxM_CB_totalEt
                        matchedIdx = idx

                if matchedIdx >= 0:
                    matched_highPt_CB_seedEta.append(_clNxM_CB_seedEta[matchedIdx])
                    matched_highPt_CB_seedPhi.append(_clNxM_CB_seedPhi[matchedIdx])
                    matched_highPt_CB_towerEm.extend(_clNxM_CB_towerEm[matchedIdx])
                    matched_highPt_CB_towerHad.extend(_clNxM_CB_towerHad[matchedIdx])
                    matched_highPt_CB_towerEgEt.extend(_clNxM_CB_towerEgEt[matchedIdx])
                    matched_highPt_CB_calibPt.append(_clNxM_CB_calibPt[matchedIdx])
                    matched_highPt_CB_IDscore.append(_clNxM_CB_IDscore[matchedIdx])

            else:
                matchedIdx = -99
                highestL1Pt = -99

                for idx in range(len(_clNxM_CE_seedEta)):
                    
                    clNxM_CE_seedEta = _clNxM_CE_seedEta[idx]
                    clNxM_CE_seedPhi = _clNxM_CE_seedPhi[idx]
                    clNxM_CE_totalEt = sum(_clNxM_CE_towerEm[idx]) + sum( _clNxM_CE_towerHad[idx]) + sum(_clNxM_CE_towerEgEt[idx])

                    dEta = gentau.Eta() - clNxM_CE_seedEta
                    dPhi = deltaPhi(gentau.Phi(), clNxM_CE_seedPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.09 and clNxM_CE_totalEt > highestL1Pt:
                        highestL1Pt = clNxM_CE_totalEt
                        matchedIdx = idx

                if matchedIdx >= 0:
                    matched_highPt_CE_seedEta.append(_clNxM_CE_seedEta[matchedIdx])
                    matched_highPt_CE_seedPhi.append(_clNxM_CE_seedPhi[matchedIdx])
                    matched_highPt_CE_towerEm.extend(_clNxM_CE_towerEm[matchedIdx])
                    matched_highPt_CE_towerHad.extend(_clNxM_CE_towerHad[matchedIdx])
                    matched_highPt_CE_towerEgEt.extend(_clNxM_CE_towerEgEt[matchedIdx])
                    matched_highPt_CE_calibPt.append(_clNxM_CE_calibPt[matchedIdx])
                    matched_highPt_CE_IDscore.append(_clNxM_CE_IDscore[matchedIdx])

                    matched_highPt_cl3d_pt.append(_cl3d_pt[matchedIdx])
                    matched_highPt_cl3d_eta.append(_cl3d_eta[matchedIdx])
                    matched_highPt_cl3d_showerlength.append(_cl3d_showerlength[matchedIdx])
                    matched_highPt_cl3d_coreshowerlength.append(_cl3d_coreshowerlength[matchedIdx])
                    matched_highPt_cl3d_spptot.append(_cl3d_spptot[matchedIdx])
                    matched_highPt_cl3d_szz.append(_cl3d_szz[matchedIdx])
                    matched_highPt_cl3d_srrtot.append(_cl3d_srrtot[matchedIdx])
                    matched_highPt_cl3d_meanz.append(_cl3d_meanz[matchedIdx])



        for tauPt, tauEta, tauPhi,  in zip(_gentau_restPt_visPt, _gentau_restPt_visEta, _gentau_restPt_visPhi):

            if tauPt < 50: continue

            gentau = ROOT.TLorentzVector()
            gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

            if abs(gentau.Eta()) > options.etaEr: continue # skip taus out of acceptance

            if abs(gentau.Eta()) < 1.5:
                matchedIdx = -99
                highestL1Pt = -99

                for idx in range(len(_clNxM_CB_seedEta)):
                    
                    clNxM_CB_seedEta = _clNxM_CB_seedEta[idx]
                    clNxM_CB_seedPhi = _clNxM_CB_seedPhi[idx]
                    clNxM_CB_totalEt = sum(_clNxM_CB_towerEm[idx]) + sum(_clNxM_CB_towerHad[idx]) + sum(_clNxM_CB_towerEgEt[idx])

                    dEta = gentau.Eta() - clNxM_CB_seedEta
                    dPhi = deltaPhi(gentau.Phi(), clNxM_CB_seedPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.09 and clNxM_CB_totalEt > highestL1Pt:
                        highestL1Pt = clNxM_CB_totalEt
                        matchedIdx = idx

                if matchedIdx >= 0:
                    matched_restPt_CB_seedEta.append(_clNxM_CB_seedEta[matchedIdx])
                    matched_restPt_CB_seedPhi.append(_clNxM_CB_seedPhi[matchedIdx])
                    matched_restPt_CB_towerEm.extend(_clNxM_CB_towerEm[matchedIdx])
                    matched_restPt_CB_towerHad.extend(_clNxM_CB_towerHad[matchedIdx])
                    matched_restPt_CB_towerEgEt.extend(_clNxM_CB_towerEgEt[matchedIdx])
                    matched_restPt_CB_calibPt.append(_clNxM_CB_calibPt[matchedIdx])
                    matched_restPt_CB_IDscore.append(_clNxM_CB_IDscore[matchedIdx])

            else:
                matchedIdx = -99
                highestL1Pt = -99

                for idx in range(len(_clNxM_CE_seedEta)):
                    
                    clNxM_CE_seedEta = _clNxM_CE_seedEta[idx]
                    clNxM_CE_seedPhi = _clNxM_CE_seedPhi[idx]
                    clNxM_CE_totalEt = sum(_clNxM_CE_towerEm[idx]) + sum( _clNxM_CE_towerHad[idx]) + sum(_clNxM_CE_towerEgEt[idx])

                    dEta = gentau.Eta() - clNxM_CE_seedEta
                    dPhi = deltaPhi(gentau.Phi(), clNxM_CE_seedPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.09 and clNxM_CE_totalEt > highestL1Pt:
                        highestL1Pt = clNxM_CE_totalEt
                        matchedIdx = idx

                if matchedIdx >= 0:
                    matched_restPt_CE_seedEta.append(_clNxM_CE_seedEta[matchedIdx])
                    matched_restPt_CE_seedPhi.append(_clNxM_CE_seedPhi[matchedIdx])
                    matched_restPt_CE_towerEm.extend(_clNxM_CE_towerEm[matchedIdx])
                    matched_restPt_CE_towerHad.extend(_clNxM_CE_towerHad[matchedIdx])
                    matched_restPt_CE_towerEgEt.extend(_clNxM_CE_towerEgEt[matchedIdx])
                    matched_restPt_CE_calibPt.append(_clNxM_CE_calibPt[matchedIdx])
                    matched_restPt_CE_IDscore.append(_clNxM_CE_IDscore[matchedIdx])

                    matched_restPt_cl3d_pt.append(_cl3d_pt[matchedIdx])
                    matched_restPt_cl3d_eta.append(_cl3d_eta[matchedIdx])
                    matched_restPt_cl3d_showerlength.append(_cl3d_showerlength[matchedIdx])
                    matched_restPt_cl3d_coreshowerlength.append(_cl3d_coreshowerlength[matchedIdx])
                    matched_restPt_cl3d_spptot.append(_cl3d_spptot[matchedIdx])
                    matched_restPt_cl3d_szz.append(_cl3d_szz[matchedIdx])
                    matched_restPt_cl3d_srrtot.append(_cl3d_srrtot[matchedIdx])
                    matched_restPt_cl3d_meanz.append(abs(_cl3d_meanz[matchedIdx]))


    gentausPt_mine = []

    matched_mine_CB_seedEta = []
    matched_mine_CB_seedPhi = []
    matched_mine_CB_towerEm = []
    matched_mine_CB_towerHad = []
    matched_mine_CB_towerEgEt = []
    matched_mine_CB_calibPt = []
    matched_mine_CB_IDscore = []
    matched_mine_CB_totalEt = []

    matched_mine_CE_seedEta = []
    matched_mine_CE_seedPhi = []
    matched_mine_CE_towerEm = []
    matched_mine_CE_towerHad = []
    matched_mine_CE_towerEgEt = []
    matched_mine_CE_calibPt = []
    matched_mine_CE_IDscore = []
    matched_mine_CE_totalEt = []

    matched_mine_cl3d_pt = []
    matched_mine_cl3d_eta = []
    matched_mine_cl3d_showerlength = []
    matched_mine_cl3d_coreshowerlength = []
    matched_mine_cl3d_spptot = []
    matched_mine_cl3d_szz = []
    matched_mine_cl3d_srrtot = []
    matched_mine_cl3d_meanz = []


    matched_mineRest_CB_seedEta = []
    matched_mineRest_CB_seedPhi = []
    matched_mineRest_CB_towerEm = []
    matched_mineRest_CB_towerHad = []
    matched_mineRest_CB_towerEgEt = []
    matched_mineRest_CB_calibPt = []
    matched_mineRest_CB_IDscore = []
    matched_mineRest_CB_totalEt = []

    matched_mineRest_CE_seedEta = []
    matched_mineRest_CE_seedPhi = []
    matched_mineRest_CE_towerEm = []
    matched_mineRest_CE_towerHad = []
    matched_mineRest_CE_towerEgEt = []
    matched_mineRest_CE_calibPt = []
    matched_mineRest_CE_IDscore = []
    matched_mineRest_CE_totalEt = []

    matched_mineRest_cl3d_pt = []
    matched_mineRest_cl3d_eta = []
    matched_mineRest_cl3d_showerlength = []
    matched_mineRest_cl3d_coreshowerlength = []
    matched_mineRest_cl3d_spptot = []
    matched_mineRest_cl3d_szz = []
    matched_mineRest_cl3d_srrtot = []
    matched_mineRest_cl3d_meanz = []

    directory = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut2p5_ptWeightedCalib/'
    inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
    inChain.Add(directory+'/Ntuple_*.root')

    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%1000==0: print('--> ',evt)
        # if evt == 200: break

        entry = inChain.GetEntry(evt)

        _gentau_visEta = list(inChain.tau_visEta)
        _gentau_visPhi = list(inChain.tau_visPhi)
        _gentau_visPt = list(inChain.tau_visPt)

        if len(_gentau_visEta) == 0: continue

        maxPt = max(_gentau_visPt)
        _gentau_visEta = [ _gentau_visEta[_gentau_visPt.index(maxPt)] ]
        _gentau_visPhi = [ _gentau_visPhi[_gentau_visPt.index(maxPt)] ]
        _gentau_visPt = [ _gentau_visPt[_gentau_visPt.index(maxPt)] ]

        gentausPt_mine.extend(_gentau_visPt)

        _clNxM_CB_seedEta = list(inChain.clNxM_CB_seedEta)
        _clNxM_CB_seedPhi = list(inChain.clNxM_CB_seedPhi)
        _clNxM_CB_towerEm = list(inChain.clNxM_CB_towerEm)
        _clNxM_CB_towerHad = list(inChain.clNxM_CB_towerHad)
        _clNxM_CB_towerEgEt = list(inChain.clNxM_CB_towerEgEt)
        _clNxM_CB_calibPt = list(inChain.clNxM_CB_calibPt)
        _clNxM_CB_IDscore = list(inChain.clNxM_CB_IDscore)

        _clNxM_CE_seedEta = list(inChain.clNxM_CE_seedEta)
        _clNxM_CE_seedPhi = list(inChain.clNxM_CE_seedPhi)
        _clNxM_CE_towerEm = list(inChain.clNxM_CE_towerEm)
        _clNxM_CE_towerHad = list(inChain.clNxM_CE_towerHad)
        _clNxM_CE_towerEgEt = list(inChain.clNxM_CE_towerEgEt)
        _clNxM_CE_calibPt = list(inChain.clNxM_CE_calibPt)
        _clNxM_CE_IDscore = list(inChain.clNxM_CE_IDscore)

        _cl3d_pt = list(inChain.cl3d_pt)
        _cl3d_eta = list(inChain.cl3d_eta)
        _cl3d_showerlength = list(inChain.cl3d_showerlength)
        _cl3d_coreshowerlength = list(inChain.cl3d_coreshowerlength)
        _cl3d_spptot = list(inChain.cl3d_spptot)
        _cl3d_szz = list(inChain.cl3d_szz)
        _cl3d_srrtot = list(inChain.cl3d_srrtot)
        _cl3d_meanz = list(inChain.cl3d_meanz)


        for tauPt, tauEta, tauPhi,  in zip(_gentau_visPt, _gentau_visEta, _gentau_visPhi):
            
            if tauPt < 50: continue

            gentau = ROOT.TLorentzVector()
            gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

            if abs(gentau.Eta()) > options.etaEr: continue # skip taus out of acceptance

            if abs(gentau.Eta()) < 1.5:
                matchedIdx = -99
                highestL1Pt = -99

                for idx in range(len(_clNxM_CB_seedEta)):
                    
                    clNxM_CB_seedEta = _clNxM_CB_seedEta[idx]
                    clNxM_CB_seedPhi = _clNxM_CB_seedPhi[idx]
                    clNxM_CB_totalEt = sum(_clNxM_CB_towerEm[idx]) + sum(_clNxM_CB_towerHad[idx]) + sum(_clNxM_CB_towerEgEt[idx])

                    dEta = gentau.Eta() - clNxM_CB_seedEta
                    dPhi = deltaPhi(gentau.Phi(), clNxM_CB_seedPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.09 and clNxM_CB_totalEt > highestL1Pt:
                        highestL1Pt = clNxM_CB_totalEt
                        matchedIdx = idx

                if matchedIdx >= 0:
                    matched_mine_CB_seedEta.append(_clNxM_CB_seedEta[matchedIdx])
                    matched_mine_CB_seedPhi.append(_clNxM_CB_seedPhi[matchedIdx])
                    matched_mine_CB_towerEm.extend(_clNxM_CB_towerEm[matchedIdx])
                    matched_mine_CB_towerHad.extend(_clNxM_CB_towerHad[matchedIdx])
                    matched_mine_CB_towerEgEt.extend(_clNxM_CB_towerEgEt[matchedIdx])
                    matched_mine_CB_calibPt.append(_clNxM_CB_calibPt[matchedIdx])
                    matched_mine_CB_IDscore.append(_clNxM_CB_IDscore[matchedIdx])

            else:
                matchedIdx = -99
                highestL1Pt = -99

                for idx in range(len(_clNxM_CE_seedEta)):
                    
                    clNxM_CE_seedEta = _clNxM_CE_seedEta[idx]
                    clNxM_CE_seedPhi = _clNxM_CE_seedPhi[idx]
                    clNxM_CE_totalEt = sum(_clNxM_CE_towerEm[idx]) + sum( _clNxM_CE_towerHad[idx]) + sum(_clNxM_CE_towerEgEt[idx])

                    dEta = gentau.Eta() - clNxM_CE_seedEta
                    dPhi = deltaPhi(gentau.Phi(), clNxM_CE_seedPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.09 and clNxM_CE_totalEt > highestL1Pt:
                        highestL1Pt = clNxM_CE_totalEt
                        matchedIdx = idx

                if matchedIdx >= 0:
                    matched_mine_CE_seedEta.append(_clNxM_CE_seedEta[matchedIdx])
                    matched_mine_CE_seedPhi.append(_clNxM_CE_seedPhi[matchedIdx])
                    matched_mine_CE_towerEm.extend(_clNxM_CE_towerEm[matchedIdx])
                    matched_mine_CE_towerHad.extend(_clNxM_CE_towerHad[matchedIdx])
                    matched_mine_CE_towerEgEt.extend(_clNxM_CE_towerEgEt[matchedIdx])
                    matched_mine_CE_calibPt.append(_clNxM_CE_calibPt[matchedIdx])
                    matched_mine_CE_IDscore.append(_clNxM_CE_IDscore[matchedIdx])

                    matched_mine_cl3d_pt.append(_cl3d_pt[matchedIdx])
                    matched_mine_cl3d_eta.append(_cl3d_eta[matchedIdx])
                    matched_mine_cl3d_showerlength.append(_cl3d_showerlength[matchedIdx])
                    matched_mine_cl3d_coreshowerlength.append(_cl3d_coreshowerlength[matchedIdx])
                    matched_mine_cl3d_spptot.append(_cl3d_spptot[matchedIdx])
                    matched_mine_cl3d_szz.append(_cl3d_szz[matchedIdx])
                    matched_mine_cl3d_srrtot.append(_cl3d_srrtot[matchedIdx])
                    matched_mine_cl3d_meanz.append(_cl3d_meanz[matchedIdx])

    
    os.system('mkdir -p MenuTausStudies')

    plt.figure(figsize=(10,10))
    plt.hist(gentausPt_highPt, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(gentausPt_restPt, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(gentausPt_mine, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('seedEta')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/gentau_visPt.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(matched_highPt_CB_seedEta, bins=np.arange(-1.55,1.55,0.05), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_seedEta, bins=np.arange(-1.55,1.55,0.05), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_seedEta, bins=np.arange(-1.55,1.55,0.05), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('seedEta')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_seedEta.pdf')
    plt.close()

    plt.hist(matched_highPt_CB_seedPhi, bins=np.arange(-3.2,3.2,0.05), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_seedPhi, bins=np.arange(-3.2,3.2,0.05), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_seedPhi, bins=np.arange(-3.2,3.2,0.05), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('seedPhi')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_seedPhi.pdf')
    plt.close()

    plt.hist(matched_highPt_CB_towerEm, bins=np.arange(0,25,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_towerEm, bins=np.arange(0,25,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_towerEm, bins=np.arange(0,25,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('towerEm')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_towerEm.pdf')
    plt.close()

    plt.hist(matched_highPt_CB_towerHad, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_towerHad, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_towerHad, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('towerHad')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_towerHad.pdf')
    plt.close()

    plt.hist(matched_highPt_CB_towerEgEt, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_towerEgEt, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_towerEgEt, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('towerEgEt')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_towerEgEt.pdf')
    plt.close()

    plt.hist(matched_highPt_CB_IDscore, bins=np.arange(0,1,0.1), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_IDscore, bins=np.arange(0,1,0.1), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_IDscore, bins=np.arange(0,1,0.1), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('IDscore')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_IDscore.pdf')
    plt.close()

    plt.hist(matched_highPt_CB_calibPt, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CB_calibPt, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CB_calibPt, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('calibPt')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CB_calibPt.pdf')
    plt.close()




    plt.figure(figsize=(10,10))
    plt.hist(matched_highPt_CE_seedEta, bins=np.arange(-3.05,3.05,0.05), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_seedEta, bins=np.arange(-3.05,3.05,0.05), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_seedEta, bins=np.arange(-3.05,3.05,0.05), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('seedEta')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_seedEta.pdf')
    plt.close()

    plt.hist(matched_highPt_CE_seedPhi, bins=np.arange(-3.2,3.2,0.05), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_seedPhi, bins=np.arange(-3.2,3.2,0.05), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_seedPhi, bins=np.arange(-3.2,3.2,0.05), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('seedPhi')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_seedPhi.pdf')
    plt.close()

    plt.hist(matched_highPt_CE_towerEm, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_towerEm, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_towerEm, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('towerEm')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_towerEm.pdf')
    plt.close()

    plt.hist(matched_highPt_CE_towerHad, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_towerHad, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_towerHad, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('towerHad')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_towerHad.pdf')
    plt.close()

    plt.hist(matched_highPt_CE_towerEgEt, bins=np.arange(0,50,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_towerEgEt, bins=np.arange(0,50,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_towerEgEt, bins=np.arange(0,50,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('towerEgEt')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_towerEgEt.pdf')
    plt.close()

    plt.hist(matched_highPt_CE_IDscore, bins=np.arange(0,1,0.1), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_IDscore, bins=np.arange(0,1,0.1), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_IDscore, bins=np.arange(0,1,0.1), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('IDscore')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_IDscore.pdf')
    plt.close()

    plt.hist(matched_highPt_CE_calibPt, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_CE_calibPt, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_CE_calibPt, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('calibPt')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/CE_calibPt.pdf')
    plt.close()


    plt.hist(matched_highPt_cl3d_pt, bins=np.arange(0,150,3), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_pt, bins=np.arange(0,150,3), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_pt, bins=np.arange(0,150,3), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('cl3d_pt')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_pt.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_eta, bins=np.arange(-3.05,3.05,0.05), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_eta, bins=np.arange(-3.05,3.05,0.05), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_eta, bins=np.arange(-3.05,3.05,0.05), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('cl3d_eta')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_eta.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_showerlength, bins=np.arange(0,50,1), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_showerlength, bins=np.arange(0,50,1), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_showerlength, bins=np.arange(0,50,1), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('cl3d_showerlength')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_showerlength.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_coreshowerlength, bins=np.arange(0,50,1), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_coreshowerlength, bins=np.arange(0,50,1), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_coreshowerlength, bins=np.arange(0,50,1), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('cl3d_coreshowerlength')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_coreshowerlength.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_spptot, bins=np.arange(0,0.075,0.001), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_spptot, bins=np.arange(0,0.075,0.001), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_spptot, bins=np.arange(0,0.075,0.001), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('cl3d_spptot')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_spptot.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_szz, bins=np.arange(0,100,1), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_szz, bins=np.arange(0,100,1), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_szz, bins=np.arange(0,100,1), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('cl3d_szz')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_szz.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_srrtot, bins=np.arange(0,0.011,0.0001), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_srrtot, bins=np.arange(0,0.011,0.0001), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_srrtot, bins=np.arange(0,0.011,0.0001), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel('cl3d_srrtot')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_srrtot.pdf')
    plt.close()

    plt.hist(matched_highPt_cl3d_meanz, bins=np.arange(320,500,5), label='Highest Pt', linewidth=2, color='green', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_restPt_cl3d_meanz, bins=np.arange(320,500,5), label='Rest Pt', linewidth=2, color='red', histtype='step', alpha=0.9, density=True)
    plt.hist(matched_mine_cl3d_meanz, bins=np.arange(320,500,5), label='Mine', linewidth=2, color='blue', histtype='step', alpha=0.9, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.yscale('log')
    plt.xlabel('cl3d_meanz')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('MenuTausStudies/cl3d_meanz.pdf')
    plt.close()











