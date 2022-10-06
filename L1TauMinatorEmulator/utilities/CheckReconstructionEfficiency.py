from array import array
import numpy as np
import ROOT
import sys
import os

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
    if delta_phi > np.pi: return 2*np.pi - delta_phi
    else:                 return delta_phi


directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v2/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT/'

inChain = ROOT.TChain("Ntuplizer/L1TauMinatorTree");
inChain.Add(directory+'/Ntuple_*.root');

etaResponse_goodNxMTaus = []
phiResponse_goodNxMTaus = []
etaResponse_badNxMTaus = []
phiResponse_badNxMTaus = []
eta_missedNxMTaus = []
phi_missedNxMTaus = []
pt_missedNxMTaus  = []

etaResponse_Cl3dTaus = []
phiResponse_Cl3dTaus = []
eta_missedCl3dTaus = []
phi_missedCl3dTaus = []
pt_missedCl3dTaus  = []

eta_mismatchedTaus = []
phi_mismatchedTaus = []
pt_mismatchedTaus = []

totalTausCnt = 0
totalEndcapTausCnt = 0
outOfAcceptanceTaus = 0
insideAcceptanceTaus = 0

goodNxMTausCnt = 0
badNxMTausCnt = 0
missedNxMTausCnt = 0

goodCl3dTausCnt = 0
missedCl3dTausCnt = 0

mismatchedTausCnt = 0


nEntries = inChain.GetEntries()
for evt in range(0, nEntries):
    if evt%1000==0: print(evt)
    # if evt == 20000: break

    entry = inChain.GetEntry(evt)

    tau_visEta = inChain.tau_visEta
    if len(tau_visEta)==0: continue
    tau_visPhi = inChain.tau_visPhi
    tau_visPt = inChain.tau_visPt
    
    _cl5x9_seedEta = list(inChain.cl5x9_seedEta)
    _cl5x9_seedPhi = list(inChain.cl5x9_seedPhi)
    _cl5x9_tauMatchIdx = list(inChain.cl5x9_tauMatchIdx)
    _cl5x9_towerEta = list(inChain.cl5x9_towerEta)
    _cl3d_eta = list(inChain.cl3d_eta)
    _cl3d_phi = list(inChain.cl3d_phi)
    _cl3d_tauMatchIdx = list(inChain.cl3d_tauMatchIdx)

    cl5x9_seedEta = []
    cl5x9_seedPhi = []
    cl5x9_tauMatchIdx = []
    cl5x9_lenVect = []
    cl3d_eta = []
    cl3d_phi = []
    cl3d_tauMatchIdx = []

    for i in [0,1]:
        try:
            idx = _cl5x9_tauMatchIdx.index(i)
            cl5x9_seedEta.append(_cl5x9_seedEta[idx])
            cl5x9_seedPhi.append(_cl5x9_seedPhi[idx])
            cl5x9_tauMatchIdx.append(_cl5x9_tauMatchIdx[idx])
            cl5x9_lenVect.append(len(_cl5x9_towerEta[idx]))
        except ValueError:
            pass

        try:
            idx = _cl3d_tauMatchIdx.index(i)
            cl3d_eta.append(_cl3d_eta[idx])
            cl3d_phi.append(_cl3d_phi[idx])
            cl3d_tauMatchIdx.append(_cl3d_tauMatchIdx[idx])
        except ValueError:
            pass

    tauIdx = 0
    for tauEta, tauPhi, tauPt in zip(tau_visEta, tau_visPhi, tau_visPt):    
        totalTausCnt += 1
        if abs(tauEta) > 3.0: outOfAcceptanceTaus += 1; tauIdx+=1; continue
        if abs(tauEta) < 3.0: insideAcceptanceTaus += 1

        ############################################
        # TowerClusters reco checks
        try:
            clNxMIdx = cl5x9_tauMatchIdx.index(tauIdx)
            clNxMEta = cl5x9_seedEta[clNxMIdx]
            clNxMPhi = cl5x9_seedPhi[clNxMIdx]
            clNxMLen = cl5x9_lenVect[clNxMIdx]

            if clNxMLen==45:
                etaResponse_goodNxMTaus.append(clNxMEta/tauEta)
                phiResponse_goodNxMTaus.append(clNxMPhi/tauPhi)
                goodNxMTausCnt += 1
            else:
                etaResponse_badNxMTaus.append(clNxMEta/tauEta)
                phiResponse_badNxMTaus.append(clNxMPhi/tauPhi)
                badNxMTausCnt += 1

        except ValueError:
            eta_missedNxMTaus.append(tauEta)
            phi_missedNxMTaus.append(tauPhi)
            pt_missedNxMTaus.append(tauPt)
            missedNxMTausCnt += 1;


        ############################################
        # Cl3D reco checks
        if abs(tauEta) > 1.5 and abs(tauEta) < 3.0: totalEndcapTausCnt += 1
        else:                                       tauIdx+=1; continue

        try:
            cl3dIdx = cl3d_tauMatchIdx.index(tauIdx)
            cl3dEta = cl3d_eta[cl3dIdx]
            cl3dPhi = cl3d_phi[cl3dIdx]

            etaResponse_Cl3dTaus.append(cl3dEta/tauEta)
            phiResponse_Cl3dTaus.append(cl3dPhi/tauPhi)
            goodCl3dTausCnt += 1

        except ValueError:
            eta_missedCl3dTaus.append(tauEta)
            phi_missedCl3dTaus.append(tauPhi)
            pt_missedCl3dTaus.append(tauPt)
            missedCl3dTausCnt += 1


        ############################################
        # Mismatched taus
        try:
            cl3dIdx = cl3d_tauMatchIdx.index(tauIdx)
            clNxMIdx = cl5x9_tauMatchIdx.index(tauIdx)

            clNxMEta = cl5x9_seedEta[clNxMIdx]
            clNxMPhi = cl5x9_seedPhi[clNxMIdx]
            clNxMLen = cl5x9_lenVect[clNxMIdx]
            cl3dEta = cl3d_eta[cl3dIdx]
            cl3dPhi = cl3d_phi[cl3dIdx]

            if abs(clNxMEta-cl3dEta)>0.25 and deltaPhi(clNxMPhi,cl3dPhi)>0.4:
                eta_mismatchedTaus.append(tauEta)
                phi_mismatchedTaus.append(tauPhi)
                pt_mismatchedTaus.append(tauPt)
                mismatchedTausCnt += 1

        except ValueError:
            pass

        tauIdx+=1


######################################################

os.system('mkdir -p RecoEffs')

plt.figure(figsize=(10,10))
plt.hist(etaResponse_goodNxMTaus, bins=np.arange(0.6,1.4,0.01), label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(etaResponse_goodNxMTaus),np.std(etaResponse_goodNxMTaus)), color='green', lw=2, histtype='step', density=True)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\eta^{L1 \tau} / \eta^{Gen \tau}$')
plt.ylabel('a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaResponseNxMTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phiResponse_goodNxMTaus, bins=np.arange(0.6,1.4,0.01), label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(phiResponse_goodNxMTaus),np.std(phiResponse_goodNxMTaus)), color='green', lw=2, histtype='step', density=True)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\phi^{L1 \tau} / \phi^{Gen \tau}$')
plt.ylabel('a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/phiResponseNxMTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(eta_missedNxMTaus, bins=np.arange(-3.1,3.1,0.1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$\eta^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaMissedNxMTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phi_missedNxMTaus, bins=np.arange(-3.2,3.2,0.1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/phiMissedNxMTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(pt_missedNxMTaus, bins=np.arange(0,200,5), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/ptMissedNxMTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(pt_missedNxMTaus, eta_missedNxMTaus, color='red')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaPtMissedNxMTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(phi_missedNxMTaus, eta_missedNxMTaus, color='red')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaPhiMissedNxMTaus.pdf')
plt.close()

######################################################

plt.figure(figsize=(10,10))
plt.hist(etaResponse_Cl3dTaus, bins=np.arange(0.9,1.1,0.005), label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(etaResponse_Cl3dTaus),np.std(etaResponse_Cl3dTaus)), color='green', lw=2, histtype='step', density=True)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\eta^{L1 \tau} / \eta^{Gen \tau}$')
plt.ylabel('a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaResponseCl3dTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phiResponse_Cl3dTaus, bins=np.arange(0.9,1.1,0.005), label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(phiResponse_Cl3dTaus),np.std(phiResponse_Cl3dTaus)), color='green', lw=2, histtype='step', density=True)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\phi^{L1 \tau} / \phi^{Gen \tau}$')
plt.ylabel('a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/phiResponseCl3dTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(eta_missedCl3dTaus, bins=np.arange(-3.1,3.1,0.1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$\eta^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaMissedCl3dTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phi_missedCl3dTaus, bins=np.arange(-3.2,3.2,0.1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/phiMissedCl3dTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(pt_missedCl3dTaus, bins=np.arange(0,200,5), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/ptMissedCl3dTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(pt_missedCl3dTaus, eta_missedCl3dTaus, color='red')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaPtMissedCl3dTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(phi_missedCl3dTaus, eta_missedCl3dTaus, color='red')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaPhiMissedCl3dTaus.pdf')
plt.close()

######################################################

plt.figure(figsize=(10,10))
plt.hist(eta_mismatchedTaus, bins=np.arange(-3.1,3.1,0.1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$\eta^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaMisMatchedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phi_mismatchedTaus, bins=np.arange(-3.2,3.2,0.1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/phiMisMatchedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(pt_mismatchedTaus, bins=np.arange(0,200,5), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/ptMisMatchedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(pt_mismatchedTaus, eta_mismatchedTaus,color='red')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaPtMisMatchedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(phi_mismatchedTaus, eta_mismatchedTaus, color='red')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs/etaPhiMisMatchedTaus.pdf')
plt.close()

######################################################

sys.stdout = Logger('RecoEffs/percentages.log')

print("")
print(" - TOTAL NUMBER OF TAUS = "+str(totalTausCnt))
print(" - TOTAL IN ACCEPTANCE  TAUS = "+str(insideAcceptanceTaus)+"  ("+str(insideAcceptanceTaus/totalTausCnt*100)+"%)")
print(" - TOTAL NUMBER OF ENDCAP TAUS = "+str(totalEndcapTausCnt)+"  ("+str(totalEndcapTausCnt/totalTausCnt*100)+"%)")
print(" - TOTAL OUT OF ACCEPTACE TAUS = "+str(outOfAcceptanceTaus)+"  ("+str(outOfAcceptanceTaus/totalTausCnt*100)+"%)")
print("")
print(" - NUMBER OF GOOD NxM TAUS = "+str(goodNxMTausCnt)+"  ("+str(goodNxMTausCnt/insideAcceptanceTaus*100)+"%)")
print(" - NUMBER OF BAD NxM TAUS = "+str(badNxMTausCnt)+"  ("+str(badNxMTausCnt/insideAcceptanceTaus*100)+"%)")
print(" - NUMBER OF MISSED NxM TAUS = "+str(missedNxMTausCnt)+"  ("+str(missedNxMTausCnt/insideAcceptanceTaus*100)+"%)")
print("")
print(" - NUMBER OF GOOD CL3D TAUS = "+str(goodCl3dTausCnt)+"  ("+str(goodCl3dTausCnt/totalEndcapTausCnt*100)+"%)")
print(" - NUMBER OF MISSED CL3D TAUS = "+str(missedCl3dTausCnt)+"  ("+str(missedCl3dTausCnt/totalEndcapTausCnt*100)+"%)")
print("")
print(" - NUMBER OF MISMATCHED TAUS = "+str(mismatchedTausCnt)+"  ("+str(mismatchedTausCnt/totalEndcapTausCnt*100)+"%)")

if insideAcceptanceTaus+outOfAcceptanceTaus != totalTausCnt: print("\n ** WARNING : total taus do not add up!")
if goodNxMTausCnt+badNxMTausCnt+missedNxMTausCnt+outOfAcceptanceTaus != totalTausCnt: print("\n ** WARNING : total calo taus do not add up!")
if goodCl3dTausCnt+missedCl3dTausCnt != totalEndcapTausCnt: print("\n ** WARNING : total endcap taus do not add up!")

# restore normal output
sys.stdout = sys.__stdout__

