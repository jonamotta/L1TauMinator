from optparse import OptionParser
from array import array
import numpy as np
import ROOT
import sys
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
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


allCltwCntVect_CB = []
allCltwCntVect_CE = []

# 1.5 is the actual geometrical separation in the simulation
# 1.55 is on tower more than the actual geometrical separation in the simulation
# 1.61 is two towers more than the actual geometrical separation in the simulation
CBCEsplits = [1.5, 1.55, 1.61]
CBCEsplittags = ['CBCEsplit1p5', 'CBCEsplit1p55', 'CBCEsplit1p61']

cltwSeedTag = 'seedEtCut2p5'

for CBCEsplit, PUtag in zip(CBCEsplits, CBCEsplittags):

    version = '3'
    user = os.getcwd().split('/')[5]
    infile_base = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+version+'/'
    inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
    # directory = infile_base+'GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut2p5/'
    # inChain.Add(directory+'/Ntuple_*.root')
    directory = infile_base+'VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_'+cltwSeedTag+'/'
    inChain.Add(directory+'/Ntuple_*.root')

    cltwCntVect_CB = []
    cltwCntVect_CE = []

    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%1000==0: print(evt)
        # if evt == 10000: break

        entry = inChain.GetEntry(evt)

        _cl3d_eta = list(inChain.cl3d_eta)
        _cl3d_phi = list(inChain.cl3d_phi)
        _cl3d_pt = list(inChain.cl3d_pt)
        _cl3d_puIdScore = list(inChain.cl3d_puidscore)
        _cl3d_stale = list(np.repeat(False, len(_cl3d_eta)))

        _cl5x9_seedEta = list(inChain.cl5x9_seedEta)
        _cl5x9_seedPhi = list(inChain.cl5x9_seedPhi)

        counter_CB = 0
        Nmatches = 0

        for cltwEta, cltwPhi in zip(_cl5x9_seedEta, _cl5x9_seedPhi):
            if abs(cltwEta) < CBCEsplit:
                counter_CB += 1
                continue

            cl3d_match = False

            idx = 0
            ptmax = -99.
            for cl3dEta, cl3dPhi, cl3dPt, cl3dPu, cl3dStale in zip(_cl3d_eta, _cl3d_phi, _cl3d_pt, _cl3d_puIdScore, _cl3d_stale):

                if cl3dPu < -0.10 or cl3dPt < 4 or cl3dStale: continue

                dEta = cltwEta - cl3dEta
                dPhi = deltaPhi(cltwPhi, cl3dPhi)
                dR2 = dEta*dEta + dPhi*dPhi

                staleIdx = -1
                if dR2 < 0.25 and cl3dPt > ptmax:
                    ptmax = cl3dPt
                    cl3d_match = True
                    staleIdx = idx

                idx += 1

            if cl3d_match:
                _cl3d_stale[staleIdx] = True
                Nmatches += 1

        cltwCntVect_CB.append(counter_CB)
        cltwCntVect_CE.append(Nmatches)

    allCltwCntVect_CB.append(cltwCntVect_CB)
    allCltwCntVect_CE.append(cltwCntVect_CE)


os.system('mkdir -p RecoEffs_vsCBCEsplit_'+cltwSeedTag)

labels = [r'CB-CE split at $\eta=1.5$', r'CB-CE split at $\eta=1.55$', r'CB-CE split at $\eta=1.61$']

cmap = get_cmap('Set1')

plt.figure(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
ymax = 0
for i, cnt in enumerate(allCltwCntVect_CB):
    hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,2))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=labels[i]+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=i+2)
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Number of clusters')
plt.ylabel('a.u.')
plt.xlim(0,50)
plt.ylim(0.0, 0.55)
# plt.xscale('symlog')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/number_of_clusters_CB.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
ymax = 0
for i, cnt in enumerate(allCltwCntVect_CE):
    hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,2))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=labels[i]+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=i+2)
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Number of clusters')
plt.ylabel('a.u.')
plt.xlim(0,50)
plt.ylim(0.0, 0.55)
# plt.xscale('symlog')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/number_of_clusters_pairs_CE.pdf')
plt.close()
