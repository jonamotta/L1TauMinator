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
    parser.add_option("--NtupleV",          dest="NtupleV",                default=None)
    parser.add_option("--v",                dest="v",                      default=None)
    parser.add_option("--date",             dest="date",                   default=None)
    parser.add_option('--etaEr',            dest='etaEr',      type=float, default=3.0)
    parser.add_option("--inTagCNN_clNxM",   dest="inTagCNN_clNxM",         default="")
    parser.add_option("--inTagDNN_cl3d",    dest="inTagDNN_cl3d",          default="")
    parser.add_option('--caloClNxM',        dest='caloClNxM',              default="5x9")
    (options, args) = parser.parse_args()
    print(options)

    os.system('mkdir -p GeomEffs')

    # working points
    # CLTW_ID_WP = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier5x9Training'+options.inTagCNN_clNxM+'/TauCNNIdentifier_plots/CLTW_TauIdentifier_WPs.pkl')
    # CL3D_ID_WP = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNIdentifierTraining'+options.inTagDNN_cl3d+'/TauDNNIdentifier_plots/CL3D_TauIdentifier_WPs.pkl')

    cl5x9_cl3d_dR = []
    cl5x9_cl3d_dEta = []
    cl5x9_cl3d_dPhi = []

    cl3d_pt_resp_closest = []
    cl3d_pt_resp_highest = []
    cl3d_pt_resp_new = []

    # cl3d_idx_closest = []
    # cl3d_idx_highest = []
    # cl3d_idx_new = []

    totalTaus = 0

    totalEndcapTaus = 0
    missedEndcapCl3d = 0
    missedEndcapCl5x9 = 0
    bothEndcapMissed = 0
    missedEndcapNew = 0

    currentGeomMatched = 0
    possibleGeomMatched = 0


    # loop over the events to fill all the histograms
    directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT/'
    inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
    inChain.Add(directory+'/Ntuple_*9*.root');
    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%1000==0: print('--> ',evt)
        if evt == 20000: break

        entry = inChain.GetEntry(evt)

        _gentau_visEta = inChain.tau_visEta
        if len(_gentau_visEta)==0: continue
        _gentau_visPhi = inChain.tau_visPhi
        _gentau_visPt = inChain.tau_visPt

        _cl5x9_pt  = list(inChain.cl5x9_calibPt)
        _cl5x9_seedEta = list(inChain.cl5x9_seedEta)
        _cl5x9_seedPhi = list(inChain.cl5x9_seedPhi)
        _cl5x9_IDscore = list(inChain.cl5x9_IDscore)
        
        _cl3d_pt  = list(inChain.cl3d_pt)
        _cl3d_eta = list(inChain.cl3d_eta)
        _cl3d_phi = list(inChain.cl3d_phi)
        _cl3d_IDscore = list(inChain.cl3d_IDscore)

        _l1tau_pt = list(inChain.minatedl1tau_pt)
        _l1tau_eta = list(inChain.minatedl1tau_eta)
        _l1tau_phi = list(inChain.minatedl1tau_phi)
        _l1tau_isBarrel = list(inChain.minatedl1tau_isBarrel)
        _l1tau_IDscore = list(inChain.minatedl1tau_IDscore)

        # cl3d_idx_closest.append([])
        # cl3d_idx_highest.append([])
        # cl3d_idx_new.append([])

        # check geometrical matching in the endcap (without any WP requirement)
        for tauPt, tauEta, tauPhi in zip(_gentau_visPt, _gentau_visEta, _gentau_visPhi):
            if abs(tauEta)<1.5 or abs(tauEta)>3.0: continue

            totalEndcapTaus += 1

            gentau = ROOT.TLorentzVector()
            gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

            # cl5x9tau = ROOT.TLorentzVector()
            cl3dtau2 = ROOT.TLorentzVector()
            cl3dtau2.SetPtEtaPhiM(0., 0., 0., 0.)

            # run on clu5x9 and store the best match
            dRmin = 0.5
            cl5x9_matched = False
            for cl5x9_pt, cl5x9_eta, cl5x9_phi, cl5x9_IDscore in zip(_cl5x9_pt, _cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_IDscore):
                if abs(cl5x9_eta) < 1.4: continue
                tmp = ROOT.TLorentzVector()
                tmp.SetPtEtaPhiM(cl5x9_pt, cl5x9_eta, cl5x9_phi, 0)

                dR = gentau.DeltaR(tmp)
                if dR <= dRmin:
                    dRmin = dR
                    cl5x9tau = tmp
                    cl5x9_matched = True

            # run on cl3d and store the best match
            dRmin = 0.5
            cl3d_matched = False
            dRfixed = 0.2
            cl3d_matched2 = False
            cl3d_idx = 0
            for cl3d_pt, cl3d_eta, cl3d_phi, cl3d_IDscore in zip(_cl3d_pt, _cl3d_eta, _cl3d_phi, _cl3d_IDscore):
                tmp = ROOT.TLorentzVector()
                tmp.SetPtEtaPhiM(cl3d_pt, cl3d_eta, cl3d_phi, 0)

                dR = gentau.DeltaR(tmp)
                if dR <= dRmin:
                    dRmin = dR
                    cl3dtau = tmp
                    cl3d_matched = True
                    # cl3d_idx_closest[evt].append(cl3d_idx)

                if dR <= dRfixed and tmp.Pt() > cl3dtau2.Pt():
                    cl3dtau2 = tmp
                    cl3d_matched2 = True
                    # cl3d_idx_highest[evt].append(cl3d_idx)

                cl3d_idx += 1

            # check if lower dR or highest pT math is best for cl3d
            if cl3d_matched:  cl3d_pt_resp_closest.append(cl3dtau.Pt()/gentau.Pt())
            if cl3d_matched2: cl3d_pt_resp_highest.append(cl3dtau2.Pt()/gentau.Pt())

            # see how many taus we are missing
            if not cl5x9_matched: missedEndcapCl5x9 += 1
            if not cl3d_matched: missedEndcapCl3d += 1
            if not cl5x9_matched and not cl3d_matched: bothEndcapMissed += 1

            # check the two possible geometrical matches
            if cl5x9_matched and cl3d_matched: 
                cl5x9_cl3d_dR.append(cl5x9tau.DeltaR(cl3dtau))
                cl5x9_cl3d_dEta.append(np.sqrt(cl5x9tau.DeltaR(cl3dtau)**2-cl5x9tau.DeltaPhi(cl3dtau)**2))
                cl5x9_cl3d_dPhi.append(cl5x9tau.DeltaPhi(cl3dtau))

                if np.sqrt(cl5x9tau.DeltaR(cl3dtau)**2-cl5x9tau.DeltaPhi(cl3dtau)**2)<0.25 and cl5x9tau.DeltaPhi(cl3dtau)<0.4:
                    currentGeomMatched += 1
                if cl5x9tau.DeltaR(cl3dtau)<0.5:
                    possibleGeomMatched += 1



            matchedNew = False
            for cl5x9_pt, cl5x9_eta, cl5x9_phi, cl5x9_IDscore in zip(_cl5x9_pt, _cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_IDscore):
                if abs(cl5x9_eta) < 1.4: continue
                cl5x9_tmp = ROOT.TLorentzVector()
                cl5x9_tmp.SetPtEtaPhiM(cl5x9_pt, cl5x9_eta, cl5x9_phi, 0)

                dRmin = 0.5
                highestPt = -99.9
                cl3d_matches_cl5x9 = False
                cl3d_idx = 0
                for cl3d_pt, cl3d_eta, cl3d_phi, cl3d_IDscore in zip(_cl3d_pt, _cl3d_eta, _cl3d_phi, _cl3d_IDscore):
                    cl3d_tmp = ROOT.TLorentzVector()
                    cl3d_tmp.SetPtEtaPhiM(cl3d_pt, cl3d_eta, cl3d_phi, 0)

                    dR = cl3d_tmp.DeltaR(cl5x9_tmp)
                    if dR <= 0.5 and cl3d_tmp.Pt() > highestPt:
                        cl3dNew = cl3d_tmp
                        highestPt = cl3d_tmp.Pt()
                        cl3d_matches_cl5x9 = True
                        # cl3d_idx_new[evt].append(cl3d_idx)

                    cl3d_idx += 1

                if cl3d_matches_cl5x9:
                # if True:
                    dRmin = 0.5
                    dR = gentau.DeltaR(cl5x9_tmp)
                    if dR <= dRmin:
                        dRmin = dR
                        cl5x9tauNew = cl5x9_tmp
                        cl3dtauNew = cl3dNew
                        matchedNew = True

            if not matchedNew: missedEndcapNew += 1
            if matchedNew: cl3d_pt_resp_new.append(cl3dtauNew.Pt()/gentau.Pt())

    # end of the loop over the events
    #################################

    print('totalEndcapTaus', totalEndcapTaus)
    print('missedEndcapCl5x9', missedEndcapCl5x9)
    print('missedEndcapCl3d', missedEndcapCl3d)
    print('bothEndcapMissed', bothEndcapMissed)
    print('currentGeomMatched', currentGeomMatched)
    print('possibleGeomMatched', possibleGeomMatched)
    print('missedEndcapNew', missedEndcapNew)

    plt.figure(figsize=(10,10))
    plt.hist(cl5x9_cl3d_dR, color='green', lw=2, histtype='step')
    plt.grid(linestyle=':')
    plt.xlabel(r'$\Delta R(CL5x9, CL3D)$')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('GeomEffs/cl5x9_cl3d_dR.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(cl5x9_cl3d_dEta, color='green', lw=2, histtype='step')
    plt.grid(linestyle=':')
    plt.xlabel(r'$\Delta\eta(CL5x9, CL3D)$')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('GeomEffs/cl5x9_cl3d_dEta.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(cl5x9_cl3d_dPhi, color='green', lw=2, histtype='step')
    plt.grid(linestyle=':')
    plt.xlabel(r'$\Delta\phi(CL5x9, CL3D)$')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('GeomEffs/cl5x9_cl3d_dPhi.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(cl3d_pt_resp_closest, bins=np.arange(0,5,0.1), color='blue', lw=2, histtype='step', label=r'Closest dR matching - $\mu$='+str(np.mean(cl3d_pt_resp_closest))+r' , $\sigma$='+str(np.std(cl3d_pt_resp_closest)))
    plt.hist(cl3d_pt_resp_highest, bins=np.arange(0,5,0.1), color='red', lw=2, histtype='step', label=r'Highest pT at fixed dR matching - $\mu$='+str(np.mean(cl3d_pt_resp_highest))+r' , $\sigma$='+str(np.std(cl3d_pt_resp_highest)))
    plt.hist(cl3d_pt_resp_new, bins=np.arange(0,5,0.1), color='green', lw=2, histtype='step', label=r'Highest pT matched to CLTW - $\mu$='+str(np.mean(cl3d_pt_resp_new))+r' , $\sigma$='+str(np.std(cl3d_pt_resp_new)))
    plt.grid(linestyle=':')
    plt.xlabel(r'$(p_T^{CL3D} / p_T^{Gen.}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc='upper right', fontsize=10)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('GeomEffs/cl3d_pt_resp_closestVShighest.pdf')
    plt.close()

