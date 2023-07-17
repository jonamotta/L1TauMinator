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


efficiencies_vsPt = []
efficiencies_vsEta = []
efficiencies_CE_vsPt = []
allCl3dCntVects = []

# 1.5 is the actual geometrical separation in the simulation
# 1.55 is on tower more than the actual geometrical separation in the simulation
# 1.61 is two towers more than the actual geometrical separation in the simulation
CBCEsplits = [1.5, 1.55, 1.61]
CBCEsplittags = ['CBCEsplit1p5', 'CBCEsplit1p55', 'CBCEsplit1p61']

cltwSeedTag = 'seedEtCut2p5'

for CBCEsplit, CBCEsplittag in zip(CBCEsplits, CBCEsplittags):

    version = '3'
    user = os.getcwd().split('/')[5]
    infile_base = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+version+'/'
    inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
    # directory = infile_base+'GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut2p5/'
    # inChain.Add(directory+'/Ntuple_*.root')
    directory = infile_base+'VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_'+cltwSeedTag+'/'
    inChain.Add(directory+'/Ntuple_*.root')

    etaResponse_CltwTaus = []
    phiResponse_CltwTaus = []
    ptResponse_CltwTaus = []

    etaResponse_Cl3dTaus = []
    phiResponse_Cl3dTaus = []
    ptResponse_Cl3dTaus = []
    puid_Cl3dTaus = []
    pionid_Cl3dTaus = []

    totalTausCnt = 0
    totalTausCnt_CE = 0

    matchedTausCnt = 0
    matchedTausCnt_CE = 0

    cltwCntVect = []

    binsPt = [25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 180, 250]
    total_vsPt = ROOT.TH1F("total_vsPt", "total_vsPt", len(binsPt)-1, array('f',binsPt))
    match_vsPt = ROOT.TH1F("match_vsPt", "match_vsPt", len(binsPt)-1, array('f',binsPt))

    total_CE_vsPt = ROOT.TH1F("total_CE_vsPt", "total_CE_vsPt", len(binsPt)-1, array('f',binsPt))
    match_CE_vsPt = ROOT.TH1F("match_CE_vsPt", "match_CE_vsPt", len(binsPt)-1, array('f',binsPt))

    binsEta = [-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    total_vsEta = ROOT.TH1F("total_vsEta", "total_vsEta", len(binsEta)-1, array('f',binsEta))
    match_vsEta = ROOT.TH1F("match_vsEta", "match_vsEta", len(binsEta)-1, array('f',binsEta))


    nEntries = inChain.GetEntries()
    for evt in range(0, nEntries):
        if evt%1000==0: print(evt)
        # if evt == 10000: break

        entry = inChain.GetEntry(evt)

        _tau_visEta = list(inChain.tau_visEta)
        _tau_visPhi = list(inChain.tau_visPhi)
        _tau_visPt = list(inChain.tau_visPt)

        _cl3d_eta = list(inChain.cl3d_eta)
        _cl3d_phi = list(inChain.cl3d_phi)
        _cl3d_pt = list(inChain.cl3d_pt)
        _cl3d_puIdScore = list(inChain.cl3d_puidscore)
        _cl3d_pionIdScore = list(inChain.cl3d_pionidscore)

        _cl5x9_seedEta = list(inChain.cl5x9_seedEta)
        _cl5x9_seedPhi = list(inChain.cl5x9_seedPhi)

        Ncl3d = 0
        for cl3dPt, cl3dPu in zip(_cl3d_pt,_cl3d_puIdScore):
            if cl3dPu < -0.10 or cl3dPt < 4: continue
            Ncl3d += 1
        cltwCntVect.append(Ncl3d)

        # first loop over taus to avoid duplicate matching
        for tauEta, tauPhi, tauPt in zip(_tau_visEta, _tau_visPhi, _tau_visPt):
            if abs(tauEta) > 3.0: continue
            if tauPt < 25: continue
            
            totalTausCnt += 1
            total_vsPt.Fill(tauPt)
            total_vsEta.Fill(tauEta)
            if abs(tauEta) < CBCEsplit:
                cltwEta_match = -99.
                cltwPhi_match = -99.

                # find best cltw match based on dR distance alone
                dR2min = 0.25
                for cltwEta, cltwPhi in zip(_cl5x9_seedEta, _cl5x9_seedPhi):

                    dEta = cltwEta - tauEta
                    dPhi = deltaPhi(cltwPhi, tauPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < dR2min:
                        dR2min = dR2

                        cltwEta_match = cltwEta
                        cltwPhi_match = cltwPhi

                if cltwEta_match == -99.: continue

                matchedTausCnt += 1

                match_vsPt.Fill(tauPt)
                match_vsEta.Fill(tauEta)

            else:
                total_CE_vsPt.Fill(tauPt)
                totalTausCnt_CE += 1

                cltwEta_match = -99.
                cltwPhi_match = -99.

                # find best cltw match based on dR distance alone
                dR2min = 0.25
                for cltwEta, cltwPhi in zip(_cl5x9_seedEta, _cl5x9_seedPhi):

                    dEta = cltwEta - tauEta
                    dPhi = deltaPhi(cltwPhi, tauPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < dR2min:
                        dR2min = dR2

                        cltwEta_match = cltwEta
                        cltwPhi_match = cltwPhi

                if cltwEta_match == -99.: continue

                cl3dEta_match = -99.
                cl3dPhi_match = -99.
                cl3dPt_match = -99.
                cl3dPu_match = -99.
                cl3dPi_match = -99.

                # find best cl3d associated to the cltw base on dR alone or on dR-pt maximization
                ptmax = -1.
                for cl3dEta, cl3dPhi, cl3dPt, cl3dPu, cl3dPi in zip(_cl3d_eta, _cl3d_phi, _cl3d_pt, _cl3d_puIdScore, _cl3d_pionIdScore):

                    if cl3dPu < -0.10 or cl3dPt < 4: continue

                    dEta = cltwEta_match - cl3dEta
                    dPhi = deltaPhi(cltwPhi_match, cl3dPhi)
                    dR2 = dEta*dEta + dPhi*dPhi

                    if dR2 < 0.25 and cl3dPt > ptmax:
                        ptmax = cl3dPt

                        cl3dEta_match = cl3dEta
                        cl3dPhi_match = cl3dPhi
                        cl3dPt_match  = cl3dPt
                        cl3dPu_match  = cl3dPu
                        cl3dPi_match  = cl3dPi

                if cl3dEta_match != -99.:
                    etaResponse_Cl3dTaus.append(tauEta - cl3dEta_match)
                    phiResponse_Cl3dTaus.append(tauPhi - cl3dPhi_match)
                    ptResponse_Cl3dTaus.append(cl3dPt_match/tauPt)
                    puid_Cl3dTaus.append(cl3dPu_match)
                    pionid_Cl3dTaus.append(cl3dPi_match)

                    matchedTausCnt_CE += 1
                    matchedTausCnt += 1

                    match_CE_vsPt.Fill(tauPt)
                    match_vsPt.Fill(tauPt)
                    match_vsEta.Fill(tauEta)


    eff_vsPt    = ROOT.TGraphAsymmErrors(match_vsPt,    total_vsPt,    "cp")
    eff_CE_vsPt = ROOT.TGraphAsymmErrors(match_CE_vsPt, total_CE_vsPt, "cp")
    eff_vsEta   = ROOT.TGraphAsymmErrors(match_vsEta,   total_vsEta,   "cp")

    efficiencies_vsPt.append(eff_vsPt)
    efficiencies_vsEta.append(eff_vsEta)
    efficiencies_CE_vsPt.append(eff_CE_vsPt)
    allCl3dCntVects.append(cltwCntVect)

    ######################################################

    os.system('mkdir -p RecoEffs_vsCBCEsplit_'+cltwSeedTag)

    sys.stdout = Logger('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/percentages_'+CBCEsplittag+'.log')

    print("")
    print(" - TOTAL NUMBER OF TAUS = "+str(totalTausCnt))
    print(" - TOTAL NUMBER OF ENDCAP TAUS = "+str(totalTausCnt_CE)+"  ("+str(totalTausCnt_CE/totalTausCnt*100)+"%)")
    print("")
    print(" - NUMBER OF MATCHED TAUS = "+str(matchedTausCnt)+"  ("+str(matchedTausCnt/totalTausCnt*100)+"%)")
    print(" - NUMBER OF ENDCAP MATCHED TAUS = "+str(matchedTausCnt_CE)+"  ("+str(matchedTausCnt_CE/totalTausCnt_CE*100)+"%)")
    print("")

    # restore normal output
    sys.stdout = sys.__stdout__

    ######################################################

    hist, edges = np.histogram(etaResponse_Cl3dTaus, bins=np.arange(-0.4,0.4,0.01))
    density = hist / len(etaResponse_Cl3dTaus)
    plt.figure(figsize=(10,10))
    plt.step(edges[1:], density, color='green', lw=2, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(etaResponse_Cl3dTaus),np.std(etaResponse_Cl3dTaus)))
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.xlabel(r'$\eta^{L1 \tau} - \eta^{Gen \tau}$')
    plt.ylabel('a.u.')
    plt.ylim(ymin=0.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/etaResponseCl3dTaus_'+CBCEsplittag+'.pdf')
    plt.close()

    hist, edges = np.histogram(phiResponse_Cl3dTaus, bins=np.arange(-0.4,0.4,0.01))
    density = hist / len(phiResponse_Cl3dTaus)
    plt.figure(figsize=(10,10))
    plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(phiResponse_Cl3dTaus),np.std(phiResponse_Cl3dTaus)), color='green', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.xlabel(r'$\phi^{L1 \tau} - \phi^{Gen \tau}$')
    plt.ylabel('a.u.')
    plt.ylim(ymin=0.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/phiResponseCl3dTaus_'+CBCEsplittag+'.pdf')
    plt.close()

    hist, edges = np.histogram(ptResponse_Cl3dTaus, bins=np.arange(-1.0,3.0,0.1))
    density = hist / len(ptResponse_Cl3dTaus)
    plt.figure(figsize=(10,10))
    plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(ptResponse_Cl3dTaus),np.std(ptResponse_Cl3dTaus)), color='green', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel('a.u.')
    plt.ylim(ymin=0.0)
    plt.xlim(0.,2.)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/ptResponseCl3dTaus_'+CBCEsplittag+'.pdf')
    plt.close()

    hist, edges = np.histogram(puid_Cl3dTaus, bins=np.arange(-1,1,0.01))
    density = hist / len(puid_Cl3dTaus)
    plt.figure(figsize=(10,10))
    plt.step(edges[1:], density, color='green', lw=2)
    plt.grid(linestyle=':')
    plt.xlabel(r'PU score')
    plt.ylabel(r'a.u.')
    plt.ylim(ymin=0.0)
    plt.xlim(0.,2.)
    plt.xlim(-1,1)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/puId_'+CBCEsplittag+'.pdf')
    plt.close()

    hist, edges = np.histogram(pionid_Cl3dTaus, bins=np.arange(-1,1,0.01))
    density = hist / len(pionid_Cl3dTaus)
    plt.figure(figsize=(10,10))
    plt.step(edges[1:], density, color='green', lw=2)
    plt.grid(linestyle=':')
    plt.xlabel(r'Pion score')
    plt.ylabel(r'a.u.')
    plt.ylim(ymin=0.0)
    plt.xlim(-1,1)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/pionId_'+CBCEsplittag+'.pdf')
    plt.close()


    ######################################################

    fig, ax = plt.subplots(figsize=(10,10))
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,eff_vsPt.GetN()):
        X.append(eff_vsPt.GetPointX(ibin))
        Y.append(eff_vsPt.GetPointY(ibin))
        Y_low.append(eff_vsPt.GetErrorYlow(ibin))
        Y_high.append(eff_vsPt.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    # leg = plt.legend(loc = 'lower right', fontsize=20)
    # leg._legend_box.align = "left"
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel('Efficiency')
    plt.xlim(0, 220)
    plt.ylim(0, 1.05)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/efficiency_vsPt_'+CBCEsplittag+'.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,eff_vsEta.GetN()):
        X.append(eff_vsEta.GetPointX(ibin))
        Y.append(eff_vsEta.GetPointY(ibin))
        Y_low.append(eff_vsEta.GetErrorYlow(ibin))
        Y_high.append(eff_vsEta.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=0.05, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    # leg = plt.legend(loc = 'lower right', fontsize=20)
    # leg._legend_box.align = "left"
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel('Efficiency')
    plt.xlim(0, 3.0)
    plt.ylim(0, 1.05)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/efficiency_vsEta_'+CBCEsplittag+'.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,eff_CE_vsPt.GetN()):
        X.append(eff_CE_vsPt.GetPointX(ibin))
        Y.append(eff_CE_vsPt.GetPointY(ibin))
        Y_low.append(eff_CE_vsPt.GetErrorYlow(ibin))
        Y_high.append(eff_CE_vsPt.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    # leg = plt.legend(loc = 'lower right', fontsize=20)
    # leg._legend_box.align = "left"
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel('Efficiency')
    plt.xlim(0, 220)
    plt.ylim(0, 1.05)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/efficiency_CE_vsPt_'+CBCEsplittag+'.pdf')
    plt.close()


    del eff_vsPt, match_vsPt, total_vsPt, eff_CE_vsPt, match_CE_vsPt, total_CE_vsPt, eff_vsEta, match_vsEta, total_vsEta


labels = [r'CB-CE split at $\eta=1.5$', r'CB-CE split at $\eta=1.55$', r'CB-CE split at $\eta=1.61$']

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
    ax.errorbar(X, Y, xerr=0.05, yerr=[Y_low, Y_high], lw=2, marker='o', color=cmap(i), zorder=i+2, label=labels[i])
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
leg = plt.legend(loc = 'lower center', fontsize=18)
leg._legend_box.align = "left"
plt.xlabel(r'$\eta_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(-3.1, 3.1)
plt.ylim(0.55, 1.01)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/efficiencies_vsEta.pdf')
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
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color=cmap(i), zorder=i+2, label=labels[i])
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
leg = plt.legend(loc = 'lower right', fontsize=20)
leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0.87, 1.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/efficiencies_vsPt.pdf')
plt.close()


fig, ax = plt.subplots(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
# plt.hlines(0.99, 0, 2000, lw=2, color='darkgray', label=r'99% efficiency', zorder=2)
# plt.hlines(0.98, 0, 2000, lw=2, color='darkgray', label=r'98% efficiency', zorder=2)
for i, currEffVsPt in enumerate(efficiencies_CE_vsPt):
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    for ibin in range(0,currEffVsPt.GetN()):
        X.append(currEffVsPt.GetPointX(ibin))
        Y.append(currEffVsPt.GetPointY(ibin))
        Y_low.append(currEffVsPt.GetErrorYlow(ibin))
        Y_high.append(currEffVsPt.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color=cmap(i), zorder=i+2, label=labels[i])
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
leg = plt.legend(loc = 'lower right', fontsize=20)
leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0.65, 1.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/efficiencies_CE_vsPt.pdf')
plt.close()


plt.figure(figsize=(10,10))
plt.grid(linestyle=':', zorder=1)
ymax = 0
for i, cnt in enumerate(allCl3dCntVects):
    hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,2))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=labels[i]+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=i+2)
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'Number of clusters')
plt.ylabel('a.u.')
plt.xlim(0,175)
plt.ylim(ymin=0.0)
# plt.xscale('symlog')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/number_of_clusters.pdf')
plt.close()


axMain = plt.subplot(111)
plt.vlines(50, 0, 1, lw=4, color='dimgray', zorder=10)
plt.grid(linestyle=':', zorder=1)
for i, cnt in enumerate(allCl3dCntVects):
    if i == 0: hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,2))
    else:      hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,1))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=labels[i]+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=i+2)
axMain.set_xscale('log')
axMain.set_xlim((50, 175))
axMain.spines['left'].set_visible(True)
axMain.yaxis.set_ticks_position('right')
axMain.yaxis.set_visible(True)
plt.setp(axMain.get_yticklabels(), visible=False)
plt.xlabel('Number of clusters')
#
divider = make_axes_locatable(axMain)
axLin = divider.append_axes("left", size=6.0, pad=0, sharey=axMain)
axLin.set_xscale('linear')
axLin.set_xlim((0, 50))
plt.grid(linestyle=':', zorder=1)
for i, cnt in enumerate(allCl3dCntVects):
    if i == 0: hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,2))
    else:      hist, edges = np.histogram(cnt, bins=np.arange(-2.5,200.5,1))
    density = hist / len(cnt)
    plt.step(edges[1:], density, label=labels[i]+r' - $\mu$=%.3f , $\sigma$=%.3f'%(np.mean(cnt),np.std(cnt)), color=cmap(i), lw=2, zorder=i+2)
axLin.spines['right'].set_visible(False)
axLin.yaxis.set_ticks_position('left')
plt.setp(axLin.get_xticklabels(), visible=True)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='')
plt.legend(loc='upper right', fontsize=16)
plt.ylim(0.0, 0.4)
plt.ylabel('a.u.')
plt.savefig('RecoEffs_vsCBCEsplit_'+cltwSeedTag+'/number_of_clusters_2.pdf')
plt.close()





