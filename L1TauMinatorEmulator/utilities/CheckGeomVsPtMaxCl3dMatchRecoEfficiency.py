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
inChain = ROOT.TChain('Ntuplizer/L1TauMinatorTree')
# directory = infile_base+'GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut2p5/'
# inChain.Add(directory+'/Ntuple_*.root')
directory = infile_base+'VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut3p5/'
inChain.Add(directory+'/Ntuple_*.root')

etaResponse_CltwTaus = []
phiResponse_CltwTaus = []
ptResponse_CltwTaus = []

etaResponse_Cl3dTaus_geom = []
phiResponse_Cl3dTaus_geom = []
ptResponse_Cl3dTaus_geom = []
puid_Cl3dTaus_geom = []
pionid_Cl3dTaus_geom = []

etaResponse_Cl3dTaus_mome = []
phiResponse_Cl3dTaus_mome = []
ptResponse_Cl3dTaus_mome = []
puid_Cl3dTaus_mome = []
pionid_Cl3dTaus_mome = []

totalTausCnt = 0
totalTausCnt_CE = 0

matchedTausCnt = 0
matchedTausCnt_CE_geom = 0
matchedTausCnt_CE_mome = 0

binsPt = [25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 180, 250]
total_vsPt = ROOT.TH1F("total_vsPt", "total_vsPt", len(binsPt)-1, array('f',binsPt))
match_geom_vsPt = ROOT.TH1F("match_geom_vsPt", "match_geom_vsPt", len(binsPt)-1, array('f',binsPt))
match_mome_vsPt = ROOT.TH1F("match_mome_vsPt", "match_mome_vsPt", len(binsPt)-1, array('f',binsPt))

total_CE_vsPt = ROOT.TH1F("total_CE_vsPt", "total_CE_vsPt", len(binsPt)-1, array('f',binsPt))
match_CE_geom_vsPt = ROOT.TH1F("match_CE_geom_vsPt", "match_CE_geom_vsPt", len(binsPt)-1, array('f',binsPt))
match_CE_mome_vsPt = ROOT.TH1F("match_CE_mome_vsPt", "match_CE_mome_vsPt", len(binsPt)-1, array('f',binsPt))

binsEta = [-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.479, -1.305, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.305, 1.479, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
total_vsEta = ROOT.TH1F("total_vsEta", "total_vsEta", len(binsEta)-1, array('f',binsEta))
match_geom_vsEta = ROOT.TH1F("match_geom_vsEta", "match_geom_vsEta", len(binsEta)-1, array('f',binsEta))
match_mome_vsEta = ROOT.TH1F("match_mome_vsEta", "match_mome_vsEta", len(binsEta)-1, array('f',binsEta))


nEntries = inChain.GetEntries()
for evt in range(0, nEntries):
    if evt%1000==0: print(evt)
    # if evt == 5000: break

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

    # first loop over taus to avoid duplicate matching
    for tauEta, tauPhi, tauPt in zip(_tau_visEta, _tau_visPhi, _tau_visPt):
        if abs(tauEta) > 3.0: continue
        if tauPt < 25: continue

        totalTausCnt += 1
        total_vsPt.Fill(tauPt)
        total_vsEta.Fill(tauEta)
        if abs(tauEta) < 1.5:
            cltwEta_geom = -99.
            cltwPhi_geom = -99.

            # find best cltw match based on dR distance alone
            dR2min = 0.25
            for cltwEta, cltwPhi in zip(_cl5x9_seedEta, _cl5x9_seedPhi):

                dEta = cltwEta - tauEta
                dPhi = deltaPhi(cltwPhi, tauPhi)
                dR2 = dEta*dEta + dPhi*dPhi

                if dR2 < dR2min:
                    dR2min = dR2

                    cltwEta_geom = cltwEta
                    cltwPhi_geom = cltwPhi

            if cltwEta_geom == -99.: continue

            matchedTausCnt += 1

            match_geom_vsPt.Fill(tauPt)
            match_mome_vsPt.Fill(tauPt)

            match_geom_vsEta.Fill(tauEta)
            match_mome_vsEta.Fill(tauEta)

        else:
            total_CE_vsPt.Fill(tauPt)
            totalTausCnt_CE += 1

            cltwEta_geom = -99.
            cltwPhi_geom = -99.

            # find best cltw match based on dR distance alone
            dR2min = 0.25
            for cltwEta, cltwPhi in zip(_cl5x9_seedEta, _cl5x9_seedPhi):

                dEta = cltwEta - tauEta
                dPhi = deltaPhi(cltwPhi, tauPhi)
                dR2 = dEta*dEta + dPhi*dPhi

                if dR2 < dR2min:
                    dR2min = dR2

                    cltwEta_geom = cltwEta
                    cltwPhi_geom = cltwPhi

            if cltwEta_geom == -99.: continue

            cl3dEta_geom = -99.
            cl3dPhi_geom = -99.
            cl3dPt_geom = -99.
            cl3dPu_geom = -99.
            cl3dPi_geom = -99.
            
            cl3dEta_mome = -99.
            cl3dPhi_mome = -99.
            cl3dPt_mome = -99.
            cl3dPu_mome = -99.
            cl3dPi_mome = -99.

            # find best cl3d associated to the cltw base on dR alone or on dR-pt maximization
            ptmax = -1.
            dR2min = 0.25
            for cl3dEta, cl3dPhi, cl3dPt, cl3dPu, cl3dPi in zip(_cl3d_eta, _cl3d_phi, _cl3d_pt, _cl3d_puIdScore, _cl3d_pionIdScore):

                if cl3dPt < 4: continue

                dEta = cltwEta_geom - cl3dEta
                dPhi = deltaPhi(cltwPhi_geom, cl3dPhi)
                dR2 = dEta*dEta + dPhi*dPhi

                if dR2 < dR2min:
                    dR2min = dR2

                    cl3dEta_geom = cl3dEta
                    cl3dPhi_geom = cl3dPhi
                    cl3dPt_geom  = cl3dPt
                    cl3dPu_geom  = cl3dPu
                    cl3dPi_geom  = cl3dPi

                if dR2 < 0.25 and cl3dPt > ptmax:
                    ptmax = cl3dPt

                    cl3dEta_mome = cl3dEta
                    cl3dPhi_mome = cl3dPhi
                    cl3dPt_mome  = cl3dPt
                    cl3dPu_mome  = cl3dPu
                    cl3dPi_mome  = cl3dPi

            if cl3dEta_geom != -99.:
                etaResponse_Cl3dTaus_geom.append(tauEta - cl3dEta_geom)
                phiResponse_Cl3dTaus_geom.append(tauPhi - cl3dPhi_geom)
                ptResponse_Cl3dTaus_geom.append(cl3dPt_geom/tauPt)
                puid_Cl3dTaus_geom.append(cl3dPu_geom)
                pionid_Cl3dTaus_geom.append(cl3dPi_geom)

                matchedTausCnt_CE_geom += 1

                match_CE_geom_vsPt.Fill(tauPt)
                match_geom_vsPt.Fill(tauPt)
                match_geom_vsEta.Fill(tauEta)

            if cl3dEta_mome != -99.:
                etaResponse_Cl3dTaus_mome.append(tauEta - cl3dEta_mome)
                phiResponse_Cl3dTaus_mome.append(tauPhi - cl3dPhi_mome)
                ptResponse_Cl3dTaus_mome.append(cl3dPt_mome/tauPt)
                puid_Cl3dTaus_mome.append(cl3dPu_mome)
                pionid_Cl3dTaus_mome.append(cl3dPi_mome)

                matchedTausCnt_CE_mome += 1

                match_CE_mome_vsPt.Fill(tauPt)
                match_mome_vsPt.Fill(tauPt)
                match_mome_vsEta.Fill(tauEta)

            if cl3dEta_geom != -99. or cl3dEta_mome != -99.:
                matchedTausCnt += 1


eff_CE_geom_vsPt = ROOT.TGraphAsymmErrors(match_CE_geom_vsPt, total_CE_vsPt, "cp")
eff_CE_mome_vsPt = ROOT.TGraphAsymmErrors(match_CE_mome_vsPt, total_CE_vsPt, "cp")
eff_geom_vsPt    = ROOT.TGraphAsymmErrors(match_geom_vsPt,    total_vsPt,    "cp")
eff_geom_vsEta   = ROOT.TGraphAsymmErrors(match_geom_vsEta,   total_vsEta,   "cp")
eff_mome_vsPt    = ROOT.TGraphAsymmErrors(match_mome_vsPt,    total_vsPt,    "cp")
eff_mome_vsEta   = ROOT.TGraphAsymmErrors(match_mome_vsEta,   total_vsEta,   "cp")

######################################################

os.system('mkdir -p GeomVsPtMaxRecoEffs')

sys.stdout = Logger('GeomVsPtMaxRecoEffs/percentages.log')

print("")
print(" - TOTAL NUMBER OF TAUS = "+str(totalTausCnt))
print(" - TOTAL NUMBER OF ENDCAP TAUS = "+str(totalTausCnt_CE))
print("")
print(" - NUMBER OF ENDCAP MATCHED TAUS (geom) = "+str(matchedTausCnt_CE_geom)+"  ("+str(matchedTausCnt_CE_geom/totalTausCnt_CE*100)+"%)")
print(" - NUMBER OF ENDCAP MATCHED TAUS (mome) = "+str(matchedTausCnt_CE_mome)+"  ("+str(matchedTausCnt_CE_mome/totalTausCnt_CE*100)+"%)")
print("")

# restore normal output
sys.stdout = sys.__stdout__

######################################################

hist, edges = np.histogram(etaResponse_Cl3dTaus_geom, bins=np.arange(-0.4,0.4,0.01))
density = hist / len(etaResponse_Cl3dTaus_geom)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(etaResponse_Cl3dTaus_geom),np.std(etaResponse_Cl3dTaus_geom)))
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\eta^{L1 \tau} - \eta^{Gen \tau}$')
plt.ylabel('a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/etaResponseCl3dTaus_geom.pdf')
plt.close()

hist, edges = np.histogram(phiResponse_Cl3dTaus_geom, bins=np.arange(-0.4,0.4,0.01))
density = hist / len(phiResponse_Cl3dTaus_geom)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(phiResponse_Cl3dTaus_geom),np.std(phiResponse_Cl3dTaus_geom)), color='green', lw=2)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\phi^{L1 \tau} - \phi^{Gen \tau}$')
plt.ylabel('a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/phiResponseCl3dTaus_geom.pdf')
plt.close()

hist, edges = np.histogram(ptResponse_Cl3dTaus_geom, bins=np.arange(-1.0,3.0,0.05))
density = hist / len(ptResponse_Cl3dTaus_geom)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(ptResponse_Cl3dTaus_geom),np.std(ptResponse_Cl3dTaus_geom)), color='green', lw=2)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
plt.ylabel('a.u.')
plt.ylim(ymin=0.0)
plt.xlim(0.,2.)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/ptResponseCl3dTaus_geom.pdf')
plt.close()

hist, edges = np.histogram(puid_Cl3dTaus_geom, bins=np.arange(-1,1,0.01))
density = hist / len(puid_Cl3dTaus_geom)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2)
plt.grid(linestyle=':')
plt.xlabel(r'PU score')
plt.ylabel(r'a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/puId_geom.pdf')
plt.close()

hist, edges = np.histogram(pionid_Cl3dTaus_geom, bins=np.arange(-1,1,0.01))
density = hist / len(pionid_Cl3dTaus_geom)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2)
plt.grid(linestyle=':')
plt.xlabel(r'Pion score')
plt.ylabel(r'a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/pionId_geom.pdf')
plt.close()



hist, edges = np.histogram(etaResponse_Cl3dTaus_mome, bins=np.arange(-0.4,0.4,0.01))
density = hist / len(etaResponse_Cl3dTaus_mome)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(etaResponse_Cl3dTaus_mome),np.std(etaResponse_Cl3dTaus_mome)), color='green', lw=2)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\eta^{L1 \tau} - \eta^{Gen \tau}$')
plt.ylabel('a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/etaResponseCl3dTaus_mome.pdf')
plt.close()

hist, edges = np.histogram(phiResponse_Cl3dTaus_mome, bins=np.arange(-0.4,0.4,0.01))
density = hist / len(phiResponse_Cl3dTaus_mome)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(phiResponse_Cl3dTaus_mome),np.std(phiResponse_Cl3dTaus_mome)), color='green', lw=2)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$\phi^{L1 \tau} - \phi^{Gen \tau}$')
plt.ylabel('a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/phiResponseCl3dTaus_mome.pdf')
plt.close()

hist, edges = np.histogram(ptResponse_Cl3dTaus_mome, bins=np.arange(-1.0,3.0,0.05))
density = hist / len(ptResponse_Cl3dTaus_mome)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, label=r'$\mu$=%.3f , $\sigma$=%.3f'%(np.mean(ptResponse_Cl3dTaus_mome),np.std(ptResponse_Cl3dTaus_mome)), color='green', lw=2)
plt.grid(linestyle=':')
plt.legend(loc = 'upper right', fontsize=16)
plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
plt.ylabel('a.u.')
plt.ylim(ymin=0.0)
plt.xlim(0.,2.)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/ptResponseCl3dTaus_mome.pdf')
plt.close()

hist, edges = np.histogram(puid_Cl3dTaus_mome, bins=np.arange(-1,1,0.01))
density = hist / len(puid_Cl3dTaus_mome)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2)
plt.grid(linestyle=':')
plt.xlabel(r'PU score')
plt.ylabel(r'a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/puId_mome.pdf')
plt.close()

hist, edges = np.histogram(pionid_Cl3dTaus_mome, bins=np.arange(-1,1,0.01))
density = hist / len(pionid_Cl3dTaus_mome)
plt.figure(figsize=(10,10))
plt.step(edges[1:], density, color='green', lw=2)
plt.grid(linestyle=':')
plt.xlabel(r'Pion score')
plt.ylabel(r'a.u.')
plt.ylim(ymin=0.0)
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/pionId_mome.pdf')
plt.close()

######################################################

fig, ax = plt.subplots(figsize=(10,10))
X = [] ; Y = [] ; Y_low = [] ; Y_high = []
for ibin in range(0,eff_geom_vsPt.GetN()):
    X.append(eff_geom_vsPt.GetPointX(ibin))
    Y.append(eff_geom_vsPt.GetPointY(ibin))
    Y_low.append(eff_geom_vsPt.GetErrorYlow(ibin))
    Y_high.append(eff_geom_vsPt.GetErrorYhigh(ibin))
ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
# leg = plt.legend(loc = 'lower right', fontsize=20)
# leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0, 1.05)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/efficiency_vsPt_geom.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
X = [] ; Y = [] ; Y_low = [] ; Y_high = []
for ibin in range(0,eff_geom_vsEta.GetN()):
    X.append(eff_geom_vsEta.GetPointX(ibin))
    Y.append(eff_geom_vsEta.GetPointY(ibin))
    Y_low.append(eff_geom_vsEta.GetErrorYlow(ibin))
    Y_high.append(eff_geom_vsEta.GetErrorYhigh(ibin))
ax.errorbar(X, Y, xerr=0.05, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
# leg = plt.legend(loc = 'lower right', fontsize=20)
# leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(-3.1, 3.1)
plt.ylim(0, 1.05)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/efficiency_vsEta_geom.pdf')
plt.close()


fig, ax = plt.subplots(figsize=(10,10))
X = [] ; Y = [] ; Y_low = [] ; Y_high = []
for ibin in range(0,eff_mome_vsPt.GetN()):
    X.append(eff_mome_vsPt.GetPointX(ibin))
    Y.append(eff_mome_vsPt.GetPointY(ibin))
    Y_low.append(eff_mome_vsPt.GetErrorYlow(ibin))
    Y_high.append(eff_mome_vsPt.GetErrorYhigh(ibin))
ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
# leg = plt.legend(loc = 'lower right', fontsize=20)
# leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0, 1.05)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/efficiency_vsPt_mome.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
X = [] ; Y = [] ; Y_low = [] ; Y_high = []
for ibin in range(0,eff_mome_vsEta.GetN()):
    X.append(eff_mome_vsEta.GetPointX(ibin))
    Y.append(eff_mome_vsEta.GetPointY(ibin))
    Y_low.append(eff_mome_vsEta.GetErrorYlow(ibin))
    Y_high.append(eff_mome_vsEta.GetErrorYhigh(ibin))
ax.errorbar(X, Y, xerr=0.05, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
# leg = plt.legend(loc = 'lower right', fontsize=20)
# leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(-3.1, 3.1)
plt.ylim(0, 1.05)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/efficiency_vsEta_mome.pdf')
plt.close()


fig, ax = plt.subplots(figsize=(10,10))
X = [] ; Y = [] ; Y_low = [] ; Y_high = []
for ibin in range(0,eff_CE_geom_vsPt.GetN()):
    X.append(eff_CE_geom_vsPt.GetPointX(ibin))
    Y.append(eff_CE_geom_vsPt.GetPointY(ibin))
    Y_low.append(eff_CE_geom_vsPt.GetErrorYlow(ibin))
    Y_high.append(eff_CE_geom_vsPt.GetErrorYhigh(ibin))
ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
# leg = plt.legend(loc = 'lower right', fontsize=20)
# leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0, 1.05)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/efficiency_CE_vsPt_geom.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
X = [] ; Y = [] ; Y_low = [] ; Y_high = []
for ibin in range(0,eff_CE_mome_vsPt.GetN()):
    X.append(eff_CE_mome_vsPt.GetPointX(ibin))
    Y.append(eff_CE_mome_vsPt.GetPointY(ibin))
    Y_low.append(eff_CE_mome_vsPt.GetErrorYlow(ibin))
    Y_high.append(eff_CE_mome_vsPt.GetErrorYhigh(ibin))
ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], lw=2, marker='o', color='green')
for xtick in ax.xaxis.get_major_ticks():
    xtick.set_pad(10)
# leg = plt.legend(loc = 'lower right', fontsize=20)
# leg._legend_box.align = "left"
plt.xlabel(r'$p_{T}^{Gen \tau}$')
plt.ylabel('Efficiency')
plt.xlim(20, 220)
plt.ylim(0, 1.05)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig('GeomVsPtMaxRecoEffs/efficiency_CE_vsPt_mome.pdf')
plt.close()

