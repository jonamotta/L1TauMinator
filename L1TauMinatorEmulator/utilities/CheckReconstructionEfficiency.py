from array import array
import numpy as np
import ROOT
import sys
import os

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v1/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT/'

inChain = ROOT.TChain("Ntuplizer/L1TauMinatorTree");
inChain.Add(directory+'/Ntuple_*.root');

etaResponse_goodTaus = []
phiResponse_goodTaus = []

etaResponse_badTaus = []
phiResponse_badTaus = []

eta_missedTaus = []
phi_missedTaus = []
pt_missedTaus  = []

totalTausCnt = 0;
goodTausCnt = 0;
badTausCnt = 0;
missedTausCnt = 0;

nEntries = inChain.GetEntries()


for i in range(0, nEntries):
    if i%1000==0: print(i)
    if i == 20000: break

    entry = inChain.GetEntry(i)

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
    cl5x9_len = []
    cl3d_eta = []
    cl3d_phi = []
    cl3d_tauMatchIdx = []

    for i in [0,1]:
        try:
            idx = _cl5x9_tauMatchIdx.index(i)
            cl5x9_seedEta.append(_cl5x9_seedEta[idx])
            cl5x9_seedPhi.append(_cl5x9_seedPhi[idx])
            cl5x9_tauMatchIdx.append(_cl5x9_tauMatchIdx[idx])
            cl5x9_len.append(len(_cl5x9_towerEta[idx]))
        except ValueError:
            pass

        try:
            idx = _cl3d_tauMatchIdx.index(i)
            cl3d_eta.append(_cl3d_eta[idx])
            cl3d_phi.append(_cl3d_phi[idx])
            cl3d_tauMatchIdx.append(_cl3d_tauMatchIdx[idx])
        except ValueError:
            pass

    usedTaus = []
    totalTausCnt += len(tau_visEta)

    for cluEta, cluPhi, cluTauIdx, cl5x9_len in zip(cl5x9_seedEta, cl5x9_seedPhi, cl5x9_tauMatchIdx, cl5x9_len):
        usedTaus.append(cluTauIdx)

        tauEta = tau_visEta[cluTauIdx]
        tauPhi = tau_visPhi[cluTauIdx]

        if cl5x9_len==45:
            etaResponse_goodTaus.append(cluEta/tauEta)
            phiResponse_goodTaus.append(cluPhi/tauPhi)
            goodTausCnt += 1
        else:
            etaResponse_badTaus.append(cluEta/tauEta)
            phiResponse_badTaus.append(cluPhi/tauPhi)
            badTausCnt += 1

        if len(usedTaus)==len(tau_visEta): break

    if len(usedTaus)!=len(tau_visEta):
        if len(usedTaus)==0:
            for i in range(len(tau_visEta)):
                eta_missedTaus.append(tau_visEta[i]);
                phi_missedTaus.append(tau_visPhi[i]);
                pt_missedTaus.append(tau_visPt[i]);
                missedTausCnt += 1;
        else:
            j = 0
            if usedTaus[0]==1: j = 1
            eta_missedTaus.append(tau_visEta[j])
            phi_missedTaus.append(tau_visPhi[j])
            pt_missedTaus.append(tau_visPt[j])
            missedTausCnt += 1

######################################################

plt.figure(figsize=(10,10))
plt.hist(etaResponse_goodTaus, bins=np.arange(0.5,1.5,0.01), label=r'Good $\tau$ leptons', color='green', lw=2, histtype='step', density=True)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
# plt.xlim(0.5,1.5)
# plt.yscale('log')
#plt.ylim(0.01,1)
plt.xlabel(r'$\eta^{L1 \tau} / \eta^{Gen \tau}$')
plt.ylabel('a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('etaResponse.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phiResponse_goodTaus, bins=np.arange(0.5,1.5,0.01), label=r'Good $\tau$ leptons', color='green', lw=2, histtype='step', density=True)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
# plt.xlim(0.5,1.5)
# plt.yscale('log')
#plt.ylim(0.01,1)
plt.xlabel(r'$\phi^{L1 \tau} / \phi^{Gen \tau}$')
plt.ylabel('a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('phiResponse.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(eta_missedTaus, bins=np.arange(-3.5,3.5,0.1), label=r'Missed $\tau$ leptons', color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.legend(loc = 'upper center', fontsize=16)
# plt.xlim(0.5,1.5)
# plt.yscale('log')
#plt.ylim(0.01,1)
plt.axvline(3.0, 0, len(eta_missedTaus)/70, color='black', lw=2)
plt.axvline(-3.0, 0, len(eta_missedTaus)/70, color='black', lw=2)
plt.xlabel(r'$\eta^{Gen \tau}$')
plt.ylabel(r'Number of $\tau$ leptons')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('etaMissedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(phi_missedTaus, bins=np.arange(-3.2,3.2,0.1), label=r'Missed $\tau$ leptons', color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
# plt.xlim(0.5,1.5)
# plt.yscale('log')
#plt.ylim(0.01,1)
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'Number of $\tau$ leptons')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('phiMissedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(pt_missedTaus, bins=np.arange(0,200,5), label=r'Missed $\tau$ leptons', color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
# plt.xlim(0.5,1.5)
#plt.ylim(0.01,1)
# plt.yscale('log')
plt.axvline(15.0, 0, len(pt_missedTaus)/40, color='black', lw=2)
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'Number of $\tau$ leptons')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('ptMissedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(pt_missedTaus, eta_missedTaus, label=r'Missed $\tau$ leptons', color='red',)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
# plt.xlim(0.5,1.5)
#plt.ylim(0.01,1)
# plt.yscale('log')
plt.axhline(3.0, 0, len(eta_missedTaus)/70, color='black', lw=2)
plt.axhline(-3.0, 0, len(eta_missedTaus)/70, color='black', lw=2)
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('etaPtMissedTaus.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(phi_missedTaus, eta_missedTaus, label=r'Missed $\tau$ leptons', color='red',)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
# plt.xlim(0.5,1.5)
#plt.ylim(0.01,1)
# plt.yscale('log')
plt.axhline(3.0, 0, len(eta_missedTaus)/70, color='black', lw=2)
plt.axhline(-3.0, 0, len(eta_missedTaus)/70, color='black', lw=2)
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'$\eta^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('etaPhiMissedTaus.pdf')
plt.close()

######################################################

print("")
print(" - TOTAL NUMBER OF TAUS = "+str(totalTausCnt))
print("")
print(" - NUMBER OF GOOD TAUS = "+str(goodTausCnt)+"  ("+str(goodTausCnt/totalTausCnt*100)+"%)")
print(" - NUMBER OF BAD TAUS = "+str(badTausCnt)+"  ("+str(badTausCnt/totalTausCnt*100)+"%)")
print(" - NUMBER OF MISSED TAUS = "+str(missedTausCnt)+"  ("+str(missedTausCnt/totalTausCnt*100)+"%)")

if goodTausCnt+badTausCnt+missedTausCnt != totalTausCnt: print("\n ** WARNING : taus do not add up!")









