from scipy.optimize import curve_fit
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

    os.system('mkdir -p WPpGeomEffs')

    # working points
    CLTW_ID_WP = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier5x9Training'+options.inTagCNN_clNxM+'/TauCNNIdentifier_plots/CLTW_TauIdentifier_WPs.pkl')
    CL3D_ID_WP = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNIdentifierTraining'+options.inTagDNN_cl3d+'/TauDNNIdentifier_plots/CL3D_TauIdentifier_WPs.pkl')

    cl5x9_cl3d_dR = []
    cl5x9_cl3d_dEta = []
    cl5x9_cl3d_dPhi = []

    cl3d_pt_resp_closest = []
    cl3d_pt_resp_highest = []

    totalTaus = 0

    goodtaus_99_99 = 0
    goodtaus_99_95 = 0
    goodtaus_99_90 = 0
    
    goodtaus_95_99 = 0
    goodtaus_95_95 = 0
    goodtaus_95_90 = 0
    
    goodtaus_90_99 = 0
    goodtaus_90_95 = 0
    goodtaus_90_90 = 0

    ptBins=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 175, 200, 500]
    offline_pts = [17.5,  22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 152.5, 157.5, 170, 187.5, 350]
    online_thresholds = range(20, 175, 1)
    plotting_thresholds = range(20, 110, 10)

    minated99_passing_ptBins = []
    minated95_passing_ptBins = []
    minated90_passing_ptBins = []
    for threshold in online_thresholds:
        minated99_passing_ptBins.append(ROOT.TH1F("minated99_passing_thr"+str(int(threshold))+"_ptBins","minated99_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
        minated95_passing_ptBins.append(ROOT.TH1F("minated95_passing_thr"+str(int(threshold))+"_ptBins","minated95_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
        minated90_passing_ptBins.append(ROOT.TH1F("minated90_passing_thr"+str(int(threshold))+"_ptBins","minated90_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))

    denominator_ptBins = ROOT.TH1F("denominator_ptBins","denominator_ptBins",len(ptBins)-1, array('f',ptBins))

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
        
        _cl3d_pt  = list(inChain.cl3d_calibPt)
        _cl3d_eta = list(inChain.cl3d_eta)
        _cl3d_phi = list(inChain.cl3d_phi)
        _cl3d_IDscore = list(inChain.cl3d_IDscore)

        _l1tau_pt = list(inChain.minatedl1tau_pt)
        _l1tau_eta = list(inChain.minatedl1tau_eta)
        _l1tau_phi = list(inChain.minatedl1tau_phi)
        _l1tau_isBarrel = list(inChain.minatedl1tau_isBarrel)
        _l1tau_IDscore = list(inChain.minatedl1tau_IDscore)

        # fill list of clusters passing the different WPs
        cl5x9s_wp99 = [] ; cl3ds_wp99 = [] ; cl3ds_wp99_score = []
        cl5x9s_wp95 = [] ; cl3ds_wp95 = [] ; cl3ds_wp95_score = []
        cl5x9s_wp90 = [] ; cl3ds_wp90 = [] ; cl3ds_wp90_score = []

        for cl5x9_pt, cl5x9_eta, cl5x9_phi, cl5x9_IDscore in zip(_cl5x9_pt, _cl5x9_seedEta, _cl5x9_seedPhi, _cl5x9_IDscore):
            cl5x9 = ROOT.TLorentzVector()
            cl5x9.SetPtEtaPhiM(cl5x9_pt, cl5x9_eta, cl5x9_phi, 0)

            if cl5x9_IDscore >= CLTW_ID_WP['wp99']: cl5x9s_wp99.append(cl5x9)
            if cl5x9_IDscore >= CLTW_ID_WP['wp95']: cl5x9s_wp95.append(cl5x9)
            if cl5x9_IDscore >= CLTW_ID_WP['wp90']: cl5x9s_wp90.append(cl5x9)

        for cl3d_pt, cl3d_eta, cl3d_phi, cl3d_IDscore in zip(_cl3d_pt, _cl3d_eta, _cl3d_phi, _cl3d_IDscore):
            cl3d = ROOT.TLorentzVector()
            cl3d.SetPtEtaPhiM(cl3d_pt, cl3d_eta, cl3d_phi, 0)

            if cl3d_IDscore >= CL3D_ID_WP['wp99']: cl3ds_wp99.append(cl3d) ; cl3ds_wp99_score.append(cl3d_IDscore)
            if cl3d_IDscore >= CL3D_ID_WP['wp95']: cl3ds_wp95.append(cl3d) ; cl3ds_wp95_score.append(cl3d_IDscore)
            if cl3d_IDscore >= CL3D_ID_WP['wp90']: cl3ds_wp90.append(cl3d) ; cl3ds_wp90_score.append(cl3d_IDscore)


        l1tau_candidates_99_99 = []
        l1tau_candidates_99_95 = []
        l1tau_candidates_99_90 = []

        l1tau_candidates_95_99 = []
        l1tau_candidates_95_95 = []
        l1tau_candidates_95_90 = []
        
        l1tau_candidates_90_99 = []
        l1tau_candidates_90_95 = []
        l1tau_candidates_90_90 = []

        # match CLTWs passing WP99 to CL3Ds passing all WP*
        for cl5x9 in cl5x9s_wp99:
            if abs(cl5x9.Eta()) > 1.5:
                
                IDmax = 0.0
                matched99 = False
                for idx, cl3d in enumerate(cl3ds_wp99):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue
                    if cl3ds_wp99_score[idx] > IDmax:
                        l1tau99 = cl3d
                        matched99 = True
                        IDmax = cl3ds_wp99_score[idx]

                IDmax = 0.0
                matched95 = False
                for idx, cl3d in enumerate(cl3ds_wp95):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue
                    if cl3ds_wp95_score[idx] > IDmax:
                        l1tau95 = cl3d
                        matched95 = True
                        IDmax = cl3ds_wp95_score[idx]

                IDmax = 0.0
                matched90 = False
                for idx, cl3d in enumerate(cl3ds_wp90):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue
                    if cl3ds_wp90_score[idx] > IDmax:
                        l1tau90 = cl3d
                        matched90 = True
                        IDmax = cl3ds_wp90_score[idx]

                if matched99: l1tau_candidates_99_99.append(l1tau99)
                if matched95: l1tau_candidates_99_95.append(l1tau95)
                if matched90: l1tau_candidates_99_90.append(l1tau90)

            else:
                l1tau_candidates_99_99.append(cl5x9)
                l1tau_candidates_99_95.append(cl5x9)
                l1tau_candidates_99_90.append(cl5x9)

        # match CLTWs passing WP95 to CL3Ds passing all WP*
        for cl5x9 in cl5x9s_wp95:
            if abs(cl5x9.Eta()) > 1.5:
                
                IDmax = 0.0
                matched99 = False
                for idx, cl3d in enumerate(cl3ds_wp99):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue
                    if cl3ds_wp99_score[idx] > IDmax:
                        l1tau99 = cl3d
                        matched99 = True
                        IDmax = cl3ds_wp99_score[idx]

                IDmax = 0.0
                matched95 = False
                for idx, cl3d in enumerate(cl3ds_wp95):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue
                    if cl3ds_wp95_score[idx] > IDmax:
                        l1tau95 = cl3d
                        matched95 = True
                        IDmax = cl3ds_wp95_score[idx]

                IDmax = 0.0
                matched90 = False
                for idx, cl3d in enumerate(cl3ds_wp90):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue
                    if cl3ds_wp90_score[idx] > IDmax:
                        l1tau90 = cl3d
                        matched90 = True
                        IDmax = cl3ds_wp90_score[idx]

                if matched99: l1tau_candidates_95_99.append(l1tau99)
                if matched95: l1tau_candidates_95_95.append(l1tau95)
                if matched90: l1tau_candidates_95_90.append(l1tau90)

            else:
                l1tau_candidates_95_99.append(cl5x9)
                l1tau_candidates_95_95.append(cl5x9)
                l1tau_candidates_95_90.append(cl5x9)

        # match CLTWs passing WP95 to CL3Ds passing all WP*
        for cl5x9 in cl5x9s_wp90:
            if abs(cl5x9.Eta()) > 1.5:
                
                IDmax = 0.0
                matched99 = False
                for idx, cl3d in enumerate(cl3ds_wp99):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue

                    if cl3ds_wp99_score[idx] > IDmax:
                        l1tau99 = cl3d
                        matched99 = True
                        IDmax = cl3ds_wp99_score[idx]

                IDmax = 0.0
                matched95 = False
                for idx, cl3d in enumerate(cl3ds_wp95):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue

                    if cl3ds_wp95_score[idx] > IDmax:
                        l1tau95 = cl3d
                        matched95 = True
                        IDmax = cl3ds_wp95_score[idx]

                IDmax = 0.0
                matched90 = False
                for idx, cl3d in enumerate(cl3ds_wp90):
                    if cl3d.DeltaR(cl5x9) > 0.5: continue

                    if cl3ds_wp90_score[idx] > IDmax:
                        l1tau90 = cl3d
                        matched90 = True
                        IDmax = cl3ds_wp90_score[idx]

                if matched99: l1tau_candidates_90_99.append(l1tau99)
                if matched95: l1tau_candidates_90_95.append(l1tau95)
                if matched90: l1tau_candidates_90_90.append(l1tau90)

            else:
                l1tau_candidates_90_99.append(cl5x9)
                l1tau_candidates_90_95.append(cl5x9)
                l1tau_candidates_90_90.append(cl5x9)


        # check out of all the candidates produced above for all the combinations of WPs which ones are properly reconstructing the taus
        for tauPt, tauEta, tauPhi in zip(_gentau_visPt, _gentau_visEta, _gentau_visPhi):
            if abs(tauEta) > options.etaEr: continue

            idx_99_99 = -99
            idx_95_95 = -99
            idx_90_90 = -99

            totalTaus += 1
            denominator_ptBins.Fill(tauPt)
            gentau = ROOT.TLorentzVector()
            gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_99_99):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_99_99 = idx
            if matched: goodtaus_99_99 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_99_95):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_99_95 = idx
            if matched: goodtaus_99_95 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_99_90):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_99_90 = idx
            if matched: goodtaus_99_90 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_95_99):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_95_99 = idx
            if matched: goodtaus_95_99 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_95_95):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_95_95 = idx
            if matched: goodtaus_95_95 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_95_90):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_95_90 = idx
            if matched: goodtaus_95_90 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_90_99):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_90_99 = idx
            if matched: goodtaus_90_99 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_90_95):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_90_95 = idx
            if matched: goodtaus_90_95 += 1

            dRmin = 0.5
            matched = False
            for idx, l1tau in enumerate(l1tau_candidates_90_90):
                if l1tau.DeltaR(gentau) < dRmin:
                    matched = True
                    dRmin = l1tau.DeltaR(gentau)
                    idx_90_90 = idx
            if matched: goodtaus_90_90 += 1


            # fill numerator histograms for every thresholds
            for i, thr in enumerate(online_thresholds): 
                if idx_99_99 != -99:
                    if l1tau_candidates_99_99[idx_99_99].Pt() > thr: minated99_passing_ptBins[i].Fill(gentau.Pt())
                if idx_95_95 != -99:
                    if l1tau_candidates_95_95[idx_95_95].Pt() > thr: minated95_passing_ptBins[i].Fill(gentau.Pt())
                if idx_90_90 != -99:
                    if l1tau_candidates_90_90[idx_90_90].Pt() > thr: minated90_passing_ptBins[i].Fill(gentau.Pt())

    # end of the loop over the events
    #################################

    # TGraphAsymmErrors for efficiency turn-ons
    turnonsMinated99 = []
    turnonsMinated95 = []
    turnonsMinated90 = []
    for i, thr in enumerate(online_thresholds):
        turnonsMinated99.append(ROOT.TGraphAsymmErrors(minated99_passing_ptBins[i], denominator_ptBins, "cp"))
        turnonsMinated95.append(ROOT.TGraphAsymmErrors(minated95_passing_ptBins[i], denominator_ptBins, "cp"))
        turnonsMinated90.append(ROOT.TGraphAsymmErrors(minated90_passing_ptBins[i], denominator_ptBins, "cp"))

    # save to file 
    fileout = ROOT.TFile("WPpGeomEffs/efficiency_graphs_er"+str(options.etaEr)+".root","RECREATE")
    denominator_ptBins.Write()
    for i, thr in enumerate(online_thresholds): 
        minated99_passing_ptBins[i].Write()
        minated95_passing_ptBins[i].Write()
        minated90_passing_ptBins[i].Write()

        turnonsMinated99[i].Write()
        turnonsMinated95[i].Write()
        turnonsMinated90[i].Write()
    fileout.Close()

    turnons_dict = {}
    for i, thr in enumerate(online_thresholds):
        turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'] = [[],[],[]]

        for ibin in range(0,turnonsMinated99[i].GetN()):
            turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0].append(turnonsMinated99[i].GetPointY(ibin))
            turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0].append(turnonsMinated95[i].GetPointY(ibin))
            turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0].append(turnonsMinated90[i].GetPointY(ibin))

            turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][1].append(turnonsMinated99[i].GetErrorYlow(ibin))
            turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][1].append(turnonsMinated95[i].GetErrorYlow(ibin))
            turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][1].append(turnonsMinated90[i].GetErrorYlow(ibin))

            turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][2].append(turnonsMinated99[i].GetErrorYhigh(ibin))
            turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][2].append(turnonsMinated95[i].GetErrorYhigh(ibin))
            turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][2].append(turnonsMinated90[i].GetErrorYhigh(ibin))



    print('totalTaus', totalTaus)
    print('goodtaus_99_99', goodtaus_99_99)
    print('goodtaus_99_95', goodtaus_99_95)
    print('goodtaus_99_90', goodtaus_99_90)
    print('goodtaus_95_99', goodtaus_95_99)
    print('goodtaus_95_95', goodtaus_95_95)
    print('goodtaus_95_90', goodtaus_95_90)
    print('goodtaus_90_99', goodtaus_90_99)
    print('goodtaus_90_95', goodtaus_90_95)
    print('goodtaus_90_90', goodtaus_90_90)


    cmap = matplotlib.cm.get_cmap('tab20c'); i=0
    ##################################################################################
    # PLOT TURNONS
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(15., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('WPpGeomEffs/turnons_WP99.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(15., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('WPpGeomEffs/turnons_WP95.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(15., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('WPpGeomEffs/turnons_WP90.pdf')
    plt.close()

