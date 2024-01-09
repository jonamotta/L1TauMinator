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


efficiencies_vsPt = []
efficiencies_vsEta = []
allCltwCntVects = []
allCltwCntVects_CE = []

seedTags = ['seedEtCut2p5', 'seedEtCut3p0', 'seedEtCut3p5', 'seedEtCut4p0', 'seedEtCut4p5', 'seedEtCut5p0']

for seedTag in seedTags:
    version = '3'
    user = os.getcwd().split('/')[5]
    infile_base = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+version+'/'
    inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
    # directory = infile_base+'GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_'+seedTag+'/'
    # inChain.Add(directory+'/Ntuple_*.root')
    directory = infile_base+'VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_'+seedTag+'/'
    inChain.Add(directory+'/Ntuple_*.root')

    etaResponse_CltwTaus = []
    phiResponse_CltwTaus = []
    ptResponse_CltwTaus = []

    totalTausCnt = 0
    matchedTausCnt= 0

    totalCltwCnt = 0
    cltwCntVect = []

    totalCltwCnt_CE = 0
    cltwCntVect_CE = []

    binsPt = [25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 180, 250]
    total_vsPt = ROOT.TH1F("total_vsPt", "total_vsPt", len(binsPt)-1, array('f',binsPt))
    match_vsPt = ROOT.TH1F("match_vsPt", "match_vsPt", len(binsPt)-1, array('f',binsPt))

    binsEta = [-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.479, -1.305, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.305, 1.479, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    total_vsEta = ROOT.TH1F("total_vsEta", "total_vsEta", len(binsEta)-1, array('f',binsEta))
    match_vsEta = ROOT.TH1F("match_vsEta", "match_vsEta", len(binsEta)-1, array('f',binsEta))

    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%1000==0: print(evt)
        if evt == 5000: break

        entry = inChain.GetEntry(evt)

        _tau_visEta = list(inChain.tau_visEta)
        _tau_visPhi = list(inChain.tau_visPhi)
        _tau_visPt = list(inChain.tau_visPt)

        _cl5x9_seedEta = list(inChain.cl5x9_seedEta)
        _cl5x9_seedPhi = list(inChain.cl5x9_seedPhi)
        _cl5x9_totalEt = list(inChain.cl5x9_totalEt)

        tmp = len(_cl5x9_seedEta)
        totalCltwCnt += tmp
        cltwCntVect.append(tmp)

        tmp1 = abs(np.array(_cl5x9_seedEta))
        tmp2 = len( tmp1[tmp1>1.5] )
        totalCltwCnt_CE += tmp2
        cltwCntVect_CE.append(tmp2)

        del tmp, tmp1, tmp2

        # first loop over taus to avoid duplicate matching
        for tauEta, tauPhi, tauPt in zip(_tau_visEta, _tau_visPhi, _tau_visPt):
            if abs(tauEta) > 3.0: continue
            if tauPt < 25: continue

            totalTausCnt += 1
            total_vsPt.Fill(tauPt)
            total_vsEta.Fill(tauEta)

            cltwEta_geom = -99.
            cltwPhi_geom = -99.
            cltwPt_geom  = -99.

            # find best cltw match based on dR distance alone
            dR2min = 0.25
            for cltwEta, cltwPhi, cltwPt in zip(_cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_totalEt):

                dEta = cltwEta - tauEta
                dPhi = deltaPhi(cltwPhi, tauPhi)
                dR2 = dEta*dEta + dPhi*dPhi

                if dR2 < dR2min:
                    dR2min = dR2

                    cltwEta_geom = cltwEta
                    cltwPhi_geom = cltwPhi
                    cltwPt_geom  = cltwPt

            if cltwEta_geom == -99.: continue

            etaResponse_CltwTaus.append(tauEta - cltwEta_geom)
            phiResponse_CltwTaus.append(deltaPhi(tauPhi, cltwPhi_geom))
            ptResponse_CltwTaus.append(cltwPt_geom/tauPt)

            matchedTausCnt += 1
            match_vsPt.Fill(tauPt)
            match_vsEta.Fill(tauEta)


    os.system('mkdir -p CltwRecoEffs')
    sys.stdout = Logger('CltwRecoEffs/percentages_'+seedTag+'.log')
    
    print("")
    print(" - TOTAL NUMBER OF TAUS = "+str(totalTausCnt))
    print(" - NUMBER OF MATCHED TAUS = "+str(matchedTausCnt)+"  ("+str(matchedTausCnt/totalTausCnt*100)+"%)")
    print(" - AVERAGE NUMBER OF CLTWs PER EVENT = "+str(totalCltwCnt/nEntries))
    print(" - AVERAGE NUMBER OF ENDCAP CLTWs PER EVENT = "+str(totalCltwCnt_CE/nEntries))
    print("")

    sys.stdout = sys.__stdout__

    eff_vsPt  = ROOT.TGraphAsymmErrors(match_vsPt,  total_vsPt,  "cp")
    eff_vsEta = ROOT.TGraphAsymmErrors(match_vsEta, total_vsEta, "cp")

    efficiencies_vsPt.append(eff_vsPt)
    efficiencies_vsEta.append(eff_vsEta)
    allCltwCntVects.append(cltwCntVect)
    allCltwCntVects_CE.append(cltwCntVect_CE)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.grid(linestyle=':', zorder=1)
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,eff_vsEta.GetN()):
        X.append(eff_vsEta.GetPointX(ibin))
        Y.append(eff_vsEta.GetPointY(ibin))
        Y_low.append(eff_vsEta.GetErrorYlow(ibin))
        Y_high.append(eff_vsEta.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=0.05, yerr=[Y_low, Y_high], lw=2, marker='o', color='green', zorder=2)
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    # leg = plt.legend(loc = 'lower right', fontsize=20)
    # leg._legend_box.align = "left"
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel('Efficiency')
    plt.xlim(-3.1, 3.1)
    plt.ylim(0.65, 1.05)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('CltwRecoEffs/efficiency_vsEta_'+seedTag+'.pdf')
    plt.close()


    fig, ax = plt.subplots(figsize=(10,10))
    plt.grid(linestyle=':', zorder=1)
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,eff_vsPt.GetN()):
        X.append(eff_vsPt.GetPointX(ibin))
        Y.append(eff_vsPt.GetPointY(ibin))
        Y_low.append(eff_vsPt.GetErrorYlow(ibin))
        Y_high.append(eff_vsPt.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green', zorder=2)
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    # leg = plt.legend(loc = 'lower right', fontsize=20)
    # leg._legend_box.align = "left"
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel('Efficiency')
    plt.xlim(20, 220)
    plt.ylim(0.75, 1.05)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('CltwRecoEffs/efficiency_vsPt_'+seedTag+'.pdf')
    plt.close()


    hist, edges = np.histogram(etaResponse_CltwTaus, bins=np.arange(-0.4,0.4,0.01))
    density = hist / len(etaResponse_CltwTaus)
    plt.figure(figsize=(10,10))
    plt.grid(linestyle=':', zorder=1)
    plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(etaResponse_CltwTaus),np.std(etaResponse_CltwTaus)), color='green', lw=2, zorder=2)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.xlabel(r'$\eta^{L1 \tau} - \eta^{Gen \tau}$')
    plt.ylabel('a.u.')
    plt.ylim(ymin=0.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('CltwRecoEffs/etaResponseCltwTaus_'+seedTag+'.pdf')
    plt.close()

    hist, edges = np.histogram(phiResponse_CltwTaus, bins=np.arange(-0.4,0.4,0.01))
    density = hist / len(phiResponse_CltwTaus)
    plt.figure(figsize=(10,10))
    plt.grid(linestyle=':', zorder=1)
    plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(phiResponse_CltwTaus),np.std(phiResponse_CltwTaus)), color='green', lw=2, zorder=2)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.xlabel(r'$\phi^{L1 \tau} - \phi^{Gen \tau}$')
    plt.ylabel('a.u.')
    plt.ylim(ymin=0.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('CltwRecoEffs/phiResponseCltwTaus_'+seedTag+'.pdf')
    plt.close()

    hist, edges = np.histogram(ptResponse_CltwTaus, bins=np.arange(0.0,3.0,0.05))
    density = hist / len(ptResponse_CltwTaus)
    plt.figure(figsize=(10,10))
    plt.grid(linestyle=':', zorder=1)
    plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(ptResponse_CltwTaus),np.std(ptResponse_CltwTaus)), color='green', lw=2, zorder=2)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel('a.u.')
    plt.ylim(ymin=0.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('CltwRecoEffs/ptResponseCltwTaus_'+seedTag+'.pdf')
    plt.close()

    del total_vsPt, match_vsPt, total_vsEta, match_vsEta, eff_vsPt, eff_vsEta, inChain


labels = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

cmap = get_cmap('Set1')

fig, ax = plt.subplots(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
for i, currEffVsEta in enumerate(efficiencies_vsEta):
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,currEffVsEta.GetN()):
        X.append(currEffVsEta.GetPointX(ibin))
        Y.append(currEffVsEta.GetPointY(ibin))
        Y_low.append(currEffVsEta.GetErrorYlow(ibin))
        Y_high.append(currEffVsEta.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=0.05, yerr=[Y_low, Y_high], lw=2, marker='o', color=cmap(i), zorder=i+2, label=r'Cluster seed $E_{T}\geq$'+'%.2f GeV'%(labels[i]))
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
leg = plt.legend(loc = 'lower right', fontsize=20)
leg._legend_box.align = "left"
plt.xlabel(r'$\eta_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(-3.1, 3.1)
plt.ylim(0.68, 1.02)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('CltwRecoEffs/efficiencies_vsEta.pdf')
plt.close()


fig, ax = plt.subplots(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
# plt.hlines(0.99, 0, 2000, lw=2, color='darkgray', label=r'99% efficiency', zorder=2)
# plt.hlines(0.98, 0, 2000, lw=2, color='darkgray', label=r'98% efficiency', zorder=2)
for i, currEffVsPt in enumerate(efficiencies_vsPt):
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,currEffVsPt.GetN()):
        X.append(currEffVsPt.GetPointX(ibin))
        Y.append(currEffVsPt.GetPointY(ibin))
        Y_low.append(currEffVsPt.GetErrorYlow(ibin))
        Y_high.append(currEffVsPt.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color=cmap(i), zorder=i+2, label=r'Cluster seed $E_{T}\geq$'+'%.2f GeV'%(labels[i]))
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
leg = plt.legend(loc = 'lower right', fontsize=20)
leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0.93, 1.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('CltwRecoEffs/efficiencies_vsPt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
ymax = 0
for i, cnt in enumerate(allCltwCntVects):
    hist, edges = np.histogram(cnt, bins=np.arange(0.5,80.5,1))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=r'Cluster seed $E_{T}\geq$'+'%.2f GeV'%(labels[i])+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=2)
    if max(density)>ymax: ymax = max(density)
plt.legend(loc = 'upper left', fontsize=16)
plt.xlabel(r'Number of clusters')
plt.ylabel('a.u.')
plt.ylim(0., ymax*1.5)
plt.xlim(10,70)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('CltwRecoEffs/number_of_clusters.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
ymax = 0
for i, cnt in enumerate(allCltwCntVects_CE):
    hist, edges = np.histogram(cnt, bins=np.arange(0.5,80.5,1))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=r'Cluster seed $E_{T}\geq$'+'%.2f GeV'%(labels[i])+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=2)
    if max(density)>ymax: ymax = max(density)
plt.legend(loc = 'upper left', fontsize=16)
plt.xlabel(r'Number of clusters')
plt.ylabel('a.u.')
plt.ylim(0., ymax*1.5)
plt.xlim(10,70)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('CltwRecoEffs/number_of_clusters_CE.pdf')
plt.close()

