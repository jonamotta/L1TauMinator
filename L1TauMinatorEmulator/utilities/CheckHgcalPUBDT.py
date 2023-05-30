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
    delta_sgn = np.sign(phi1-phi2)
    if delta_phi > np.pi: return delta_sgn * (2*np.pi - delta_phi)
    else:                 return delta_sgn * delta_phi


version = '3'
user = os.getcwd().split('/')[5]
infile_base = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+version+'/'
inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
# directory = infile_base+'GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut2p5/'
# inChain.Add(directory+'/Ntuple_*.root')
directory = infile_base+'VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut3p5/'
inChain.Add(directory+'/Ntuple_*.root')

goodCl3d_puscore = []
badCl3d_puscore = []
goodCl3d_pionscore = []
badCl3d_pionscore = []

tau_pt = []
goodCl3d_pt = []
badCl3d_pt = []

nEntries = inChain.GetEntries()
for evt in range(0, nEntries):
    if evt%1000==0: print(evt)
    # if evt == 2000: break

    entry = inChain.GetEntry(evt)

    _tau_visEta = list(inChain.tau_visEta)
    _tau_visPhi = list(inChain.tau_visPhi)
    _tau_visPt = list(inChain.tau_visPt)

    _cl3d_eta = list(inChain.cl3d_eta)
    _cl3d_phi = list(inChain.cl3d_phi)
    _cl3d_pt = list(inChain.cl3d_pt)
    _cl3d_puIdScore = list(inChain.cl3d_puidscore)
    _cl3d_pionIdScore = list(inChain.cl3d_pionidscore)
    _cl3d_tauMatchIdx = list(inChain.cl3d_tauMatchIdx)

    for idx in range(len(_cl3d_eta)):
        if _cl3d_tauMatchIdx[idx] != -99:
            goodCl3d_puscore.append(_cl3d_puIdScore[idx])
            goodCl3d_pionscore.append(_cl3d_pionIdScore[idx])
        
            tau_pt.append(_tau_visPt[_cl3d_tauMatchIdx[idx]])
            goodCl3d_pt.append(_cl3d_pt[idx])
        
        else:
            badCl3d_puscore.append(_cl3d_puIdScore[idx])
            badCl3d_pionscore.append(_cl3d_pionIdScore[idx])
        
            badCl3d_pt.append(_cl3d_pt[idx])


os.system('mkdir -p BasicHgcalPuBdtPlots')

plt.figure(figsize=(10,10))
plt.hist(goodCl3d_puscore, bins=np.arange(-1,1,0.01), density=True, color='green', lw=2, histtype='step')
plt.hist(badCl3d_puscore, bins=np.arange(-1,1,0.01), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'PU score')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/puId.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(goodCl3d_pionscore, bins=np.arange(-1,1,0.01), density=True, color='green', lw=2, histtype='step')
plt.hist(badCl3d_pionscore, bins=np.arange(-1,1,0.01), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'Pion score')
plt.ylabel(r'a.u.')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/pionId.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.hist(goodCl3d_pt, bins=np.arange(0,100,1), density=True, color='green', lw=2, histtype='step')
plt.hist(badCl3d_pt, bins=np.arange(0,100,1), density=True, color='red', lw=2, histtype='step')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{CL3D}$')
plt.ylabel(r'a.u.')
plt.yscale("log")
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/pt.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(badCl3d_puscore[:5000], badCl3d_pt[:5000], color='red', alpha=0.5)
plt.scatter(goodCl3d_puscore, goodCl3d_pt, color='green', alpha=0.5)
plt.grid(linestyle=':')
plt.xlabel(r'PU score')
plt.ylabel(r'$p_{T}^{CL3D}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/cl3dpt_vs_puscore.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(badCl3d_pionscore[:5000], badCl3d_pt[:5000], color='red', alpha=0.5)
plt.scatter(goodCl3d_pionscore, goodCl3d_pt, color='green', alpha=0.5)
plt.grid(linestyle=':')
plt.xlabel(r'Pion score')
plt.ylabel(r'$p_{T}^{CL3D}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/cl3dpt_vs_pionscore.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(goodCl3d_puscore, tau_pt, color='green', alpha=0.5, )
plt.grid(linestyle=':')
plt.xlabel(r'PU score')
plt.ylabel(r'$p_{T}^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/taupt_vs_puscore.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.scatter(goodCl3d_pionscore, tau_pt, color='green', alpha=0.5, )
plt.grid(linestyle=':')
plt.xlabel(r'Pion score')
plt.ylabel(r'$p_{T}^{Gen \tau}$')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/taupt_vs_pionscore.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(badCl3d_pionscore[:5000], badCl3d_puscore[:5000], color='red', alpha=0.5)
plt.scatter(goodCl3d_pionscore, goodCl3d_puscore, color='green', alpha=0.5)
plt.grid(linestyle=':')
plt.xlabel(r'Pion score')
plt.ylabel(r'PU score')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicHgcalPuBdtPlots/puscore_vs_pionscore.pdf')
plt.close()







