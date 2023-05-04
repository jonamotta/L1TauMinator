from optparse import OptionParser
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
directory = infile_base+'GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD/'
inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
inChain.Add(directory+'/Ntuple_*.root')

seedTowerEt_CltwTaus = []
seedTowerEm_CltwTaus = []
seedTowerHad_CltwTaus = []
seedTowerEgEt_CltwTaus = []
tausEtas = []
tausPhis = []
tausPts = []

idxi = 0

nEntries = inChain.GetEntries()
for evt in range(0, nEntries):
    if evt%1000==0: print(evt)
    if evt == 5000: break

    entry = inChain.GetEntry(evt)

    _tau_visEta = list(inChain.tau_visEta)
    _tau_visPhi = list(inChain.tau_visPhi)
    _tau_visPt  = list(inChain.tau_visPt)

    _cl5x9_seedEta = list(inChain.cl5x9_seedEta)
    _cl5x9_seedPhi = list(inChain.cl5x9_seedPhi)
    _cl5x9_towerEt = list(inChain.cl5x9_towerEt)
    _cl9x9_towerEm  = list(inChain.cl9x9_towerEm)
    _cl9x9_towerHad = list(inChain.cl9x9_towerHad)
    _cl9x9_towerEgEt = list(inChain.cl9x9_towerEgEt)
    _cl5x9_towerIEta = list(inChain.cl5x9_towerEta)
    _cl5x9_towerIPhi = list(inChain.cl5x9_towerPhi)

    # first loop over taus to avoid duplicate matching
    for tauEta, tauPhi, tauPt in zip(_tau_visEta, _tau_visPhi, _tau_visPt):
        if abs(tauEta) > 3.0: continue
        if tauPt < 15: continue

        cltwEta_geom = -99.
        cltwPhi_geom = -99.
        cltwSeedEt_geom  = -99.
        cltwSeedEm_geom  = -99.
        cltwSeedHad_geom  = -99.
        cltwSeedEgEt_geom  = -99.
        cltwTowIEta_geom  = -99.
        cltwTowIPhi_geom  = -99.

        # find best cltw match based on dR distance alone
        dR2min = 0.25
        for cltwEta, cltwPhi, cltwTEt, cltwTEm, cltwTHad, cltwTEgEt, cltwTIEta, cltwTIPhi in zip(_cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_towerEt, _cl9x9_towerEm, _cl9x9_towerHad, _cl9x9_towerEgEt, _cl5x9_towerIEta, _cl5x9_towerIPhi):

            if cltwTEt[22] < 2.5:
                idxi += 1
                
                print('cluster'+str(idxi)+'_broken = np.array([')
                for eta, phi in zip(cltwTIEta, cltwTIPhi):
                    print('['+str(eta)+','+str(phi)+'],')
                print('])')

            dEta = cltwEta - tauEta
            dPhi = deltaPhi(cltwPhi, tauPhi)
            dR2 = dEta*dEta + dPhi*dPhi

            if dR2 < dR2min:
                dR2min = dR2

                cltwEta_geom = cltwEta
                cltwPhi_geom = cltwPhi
                cltwSeedEt_geom = cltwTEt[22]
                cltwSeedEm_geom = cltwTEm[22]
                cltwSeedHad_geom = cltwTHad[22]
                cltwSeedEgEt_geom = cltwTHad[22]
                # cltwTowIEta_geom = 
                # cltwTowIPhi_geom = 

        if cltwEta_geom == -99.: continue

        seedTowerEt_CltwTaus.append(cltwSeedEt_geom)
        seedTowerEm_CltwTaus.append(cltwSeedEm_geom)
        seedTowerHad_CltwTaus.append(cltwSeedHad_geom)
        seedTowerEgEt_CltwTaus.append(cltwSeedEgEt_geom)
        tausEtas.append(tauEta)
        tausPhis.append(tauPhi)
        tausPts.append(tauPt)


# print(np.min(seedTowerEt_CltwTaus))

os.system('mkdir -p BasicSeedingPlots')

hist, edges = np.histogram(seedTowerEt_CltwTaus, bins=np.arange(0,100,1))
density = hist / len(seedTowerEt_CltwTaus)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(seedTowerEt_CltwTaus),np.std(seedTowerEt_CltwTaus)))
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Seed TT $E_{T}$')
plt.ylabel('a.u.')
plt.yscale('log')
plt.xlim(1,40)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEt.pdf')
plt.close()


hist, edges = np.histogram(seedTowerEm_CltwTaus, bins=np.arange(0,100,1))
density = hist / len(seedTowerEm_CltwTaus)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(seedTowerEm_CltwTaus),np.std(seedTowerEm_CltwTaus)))
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Seed TT $E_{T}$')
plt.ylabel('a.u.')
plt.yscale('log')
plt.xlim(1,20)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEm.pdf')
plt.close()


hist, edges = np.histogram(seedTowerHad_CltwTaus, bins=np.arange(0,100,1))
density = hist / len(seedTowerHad_CltwTaus)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(seedTowerHad_CltwTaus),np.std(seedTowerHad_CltwTaus)))
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Seed TT $E_{T}$')
plt.ylabel('a.u.')
plt.yscale('log')
plt.xlim(1,20)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerHad.pdf')
plt.close()

hist, edges = np.histogram(seedTowerEgEt_CltwTaus, bins=np.arange(0,100,1))
density = hist / len(seedTowerEgEt_CltwTaus)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(seedTowerEgEt_CltwTaus),np.std(seedTowerEgEt_CltwTaus)))
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Seed TT $E_{T}$')
plt.ylabel('a.u.')
plt.yscale('log')
plt.xlim(1,20)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEgEt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(tausPts, seedTowerEt_CltwTaus, color='green', marker='o')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'Seed TT $E_{T}$')
plt.ylim(ymin=0.0)
# plt.yscale('log')
plt.ylim(1,40)
plt.xlim(14,75)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEt_vs_TauPt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(tausPts, seedTowerEm_CltwTaus, color='green', marker='o')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'Seed TT $E_{T}$')
plt.ylim(ymin=0.0)
# plt.yscale('log')
plt.ylim(0,10)
plt.xlim(14,75)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEm_vs_TauPt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(tausPts, seedTowerHad_CltwTaus, color='green', marker='o')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'Seed TT $E_{T}$')
plt.ylim(ymin=0.0)
# plt.yscale('log')
plt.ylim(0,10)
plt.xlim(14,75)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerHad_vs_TauPt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(tausPts, seedTowerEgEt_CltwTaus, color='green', marker='o')
plt.grid(linestyle=':')
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel(r'Seed TT $E_{T}$')
plt.ylim(ymin=0.0)
# plt.yscale('log')
plt.ylim(0,10)
plt.xlim(14,75)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEgEt_vs_TauPt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(tausEtas, seedTowerEt_CltwTaus, color='green', marker='o')
plt.grid(linestyle=':')
plt.xlabel(r'$\eta^{Gen \tau}$')
plt.ylabel(r'Seed TT $E_{T}$')
plt.ylim(ymin=0.0)
# plt.yscale('log')
plt.ylim(1,40)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEt_vs_TauEta.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.scatter(tausPhis, seedTowerEt_CltwTaus, color='green', marker='o')
plt.grid(linestyle=':')
plt.xlabel(r'$\phi^{Gen \tau}$')
plt.ylabel(r'Seed TT $E_{T}$')
plt.ylim(ymin=0.0)
# plt.yscale('log')
plt.ylim(1,40)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('BasicSeedingPlots/seedTowerEt_vs_TauPhi.pdf')
plt.close()







