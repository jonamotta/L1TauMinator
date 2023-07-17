from scipy.optimize import curve_fit
from scipy.signal import convolve
from optparse import OptionParser
from scipy import interpolate
import scipy.special as sp
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

################################################################
## MENU WAY OF DOING SCALING

def _interpolate(H, K1, K2):

        A = np.ones(len(K1)) * (-K2)
        B = [k1i + 2 * K2 for k1i in K1]
        C = np.ones(len(K1)) * (-K2)
        D = []

        A[0] = 0
        C[-1] = 0

        D = [k1i * h1i for k1i, h1i in zip(K1, H)]

        D[0] = D[0] + K2 * 0
        D[-1] = D[-1] + K2 * 1

        for i in range(1, len(K1)):
            F = A[i] / B[i - 1]

            A[i] = A[i] - B[i - 1] * F
            B[i] = B[i] - C[i - 1] * F
            C[i] = C[i]
            D[i] = D[i] - D[i - 1] * F

        Y = np.ones(len(K1))
        Y[-1] = D[-1] / B[-1]

        for i in reversed(range(len(K1) - 2)):
            Y[i] = (D[i] - C[i] * Y[i + 1]) / B[i]

        return Y

def _get_point_on_curve(x, graph_x, graph_y):

        if (x < graph_x[0]):
            return 0

        if (x >= graph_x[len(graph_x) - 1]):
            return 1

        xr = graph_x[0]
        yr = graph_y[0]
        for i in range(len(graph_x) - 1):
            xl = xr
            yl = yr
            xr = graph_x[i + 1]
            yr = graph_y[i + 1]
            if ((x < xr) & (x >= xl)):
                return yl + (yr - yl) / (xr - xl) * (x - xl)

        return -1

def _find_turnon_cut(graph_x, graph_y, Target):
        L = 0
        R = np.max(graph_x)

        while (R - L > 0.0001):
            C = (L + R) / 2
            V = _get_point_on_curve(C, graph_x, graph_y)

            if (V < Target):
                L = C
            else:
                R = C

        return (R + L) / 2.


################################################################
## MY WAY OF DOING SCALING

def MYinterpolate(x, y, eff):
        spline = interpolate.Akima1DInterpolator(x, np.array(y) - eff)
        roots = spline.roots()
        
        # in case there is no root just return the maximum x value
        if len(roots) == 0:
            return max(x)
        else:
            # this for loop makes sure to take only positive
            # roots and skip spurious negative roots
            for root in roots:
                if root > 0:
                    return root

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option("--NtupleV",          dest="NtupleV",        default=None)
    parser.add_option("--v",                dest="v",              default=None)
    parser.add_option("--date",             dest="date",           default=None)
    parser.add_option('--loop',             dest='loop',           default=False, action="store_true")
    parser.add_option('--etaEr',            dest='etaEr',          default=3.0,   type=float)
    parser.add_option('--WP_CB',            dest='WP_CB',          default='90')
    parser.add_option('--WP_CE',            dest='WP_CE',          default='90')
    parser.add_option('--CBCEsplit',        dest='CBCEsplit',      default=1.55, type=float)
    parser.add_option('--caloClNxM',        dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",        dest="seedEtCut",      default="2p5")
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    ptBins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 175, 200, 500]
    offline_pts = [2.5, 7.5, 12.5, 17.5,  22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 152.5, 157.5, 170, 187.5, 350]

    # MENU BINNING
    # ptBins = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150]
    # offline_pts = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 99, 105, 111, 117, 123, 129, 135, 141, 147]

    if options.etaEr==3.0:
        etaBins=[-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
        eta_bins_centers = [-2.95, -2.85, -2.75, -2.65, -2.55, -2.45, -2.35, -2.25, -2.15, -2.05, -1.95, -1.85, -1.75, -1.65, -1.55, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95]
    elif options.etaEr==2.4:
        etaBins = [-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,  2.2, 2.4]
        eta_bins_centers = [-2.3, -2.1, -1.9, -1.7, -1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1,  0.1,  0.3,  0.5,  0.7,  0.9,  1.1,  1.3,  1.5,  1.7,  1.9, 2.1,  2.3]

    else:
        exit()

    online_thresholds = range(0, 175, 1)
    plotting_thresholds = range(10, 110, 10)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag+'_SQUARE'
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/turnons'+tag)

    WP_CB = options.WP_CB
    WP_CE = options.WP_CE
    qualityCut_CB = 0
    qualityCut_CE = 0
    if WP_CB == "99": qualityCut_CB = 1
    if WP_CB == "95": qualityCut_CB = 2
    if WP_CB == "90": qualityCut_CB = 3
    if WP_CE == "99": qualityCut_CE = 1
    if WP_CE == "95": qualityCut_CE = 2
    if WP_CE == "90": qualityCut_CE = 3

    if options.loop:
        #passing histograms (numerators)
        minated_passing_ptBins = []
        minated_passing_ptBins_CB = []
        minated_passing_ptBins_CE = []

        minated_passing_etaBins = []

        for threshold in online_thresholds:
            minated_passing_ptBins.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_ptBins","minated_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated_passing_ptBins_CB.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_ptBins_CB","minated_passing_thr"+str(int(threshold))+"_ptBins_CB",len(ptBins)-1, array('f',ptBins)))
            minated_passing_ptBins_CE.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_ptBins_CE","minated_passing_thr"+str(int(threshold))+"_ptBins_CE",len(ptBins)-1, array('f',ptBins)))
            
            minated_passing_etaBins.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_etaBins","minated_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))

        #denominator
        denominator_ptBins = ROOT.TH1F("denominator_ptBins","denominator_ptBins",len(ptBins)-1, array('f',ptBins))
        denominator_ptBins_CB = ROOT.TH1F("denominator_ptBins_CB","denominator_ptBins_CB",len(ptBins)-1, array('f',ptBins))
        denominator_ptBins_CE = ROOT.TH1F("denominator_ptBins_CE","denominator_ptBins_CE",len(ptBins)-1, array('f',ptBins))
        denominator_etaBins = ROOT.TH1F("denominator_etaBins","denominator_etaBins",len(etaBins)-1, array('f',etaBins))

        tot = 0

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        if "PRD" in options.inTag:   inChain = ROOT.TChain("L1CaloTauNtuplizerProducer/L1TauMinatorTree");
        elif "EMU" in options.inTag: inChain = ROOT.TChain("L1CaloTauNtuplizerEmulator/L1TauMinatorTree");
        else:                        inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");

        inChain.Add(directory+'/Ntuple_*.root');
        nEntries = inChain.GetEntries()
        for evt in range(0, nEntries):
            if evt%1000==0: print('--> ',evt)
            # if evt == 10000: break

            entry = inChain.GetEntry(evt)

            ### MENU'S TAUS
            _gentau_visEta = list(inChain.tau_visEta)
            _gentau_visPhi = list(inChain.tau_visPhi)
            _gentau_visPt = list(inChain.tau_visPt)

            _l1tau_pt = list(inChain.squarel1tau_pt)
            _l1tau_eta = list(inChain.squarel1tau_eta)
            _l1tau_phi = list(inChain.squarel1tau_phi)
            _l1tau_quality = list(inChain.squarel1tau_isoEt)
            # _l1tau_quality = list(inChain.minatedl1tau_IDscore)

            if len(_gentau_visPt) == 0: continue

            ###########################################################################
            # BARREL TURNON
            maxPt = -99.
            for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
                if abs(genEta) > 1.5: continue
                if genPt > maxPt: maxPt = genPt

            if maxPt > 0:
                maxEta = _gentau_visEta[_gentau_visPt.index(maxPt)]
                maxPhi = _gentau_visPhi[_gentau_visPt.index(maxPt)]

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(maxPt, maxEta, maxPhi, 0)

                if abs(gentau.Eta()) < options.etaEr: # skip taus out of acceptance

                    denominator_ptBins_CB.Fill(gentau.Pt())

                    minatedMatched = False
                    highestMinatedL1Pt = -99.9

                    # loop over TauMinator taus
                    for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_quality):

                        l1tau = ROOT.TLorentzVector()
                        l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                        if abs(l1tau.Eta()) > options.etaEr: continue # skip taus out of acceptance
                        #if abs(l1tau.Eta()) < options.CBCEsplit:
                        #    if l1tauId < qualityCut_CB: continue
                        #else:
                        #    if l1tauId < qualityCut_CE: continue

                        # check matching
                        if gentau.DeltaR(l1tau)<0.3:
                            # keep only L1 match with highest pT
                            if l1tau.Pt()>highestMinatedL1Pt:
                                minatedMatched = True
                                highestMinatedL1Pt = l1tau.Pt()

                    # fill numerator histograms for every thresholds
                    for i, thr in enumerate(online_thresholds): 
                        if minatedMatched and highestMinatedL1Pt>float(thr):
                            minated_passing_ptBins_CB[i].Fill(gentau.Pt())


            ###########################################################################
            # ENDCAP TURNON
            maxPt = -99.
            for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
                if abs(genEta) < 1.5: continue
                if genPt > maxPt: maxPt = genPt

            if maxPt > 0:
                maxEta = _gentau_visEta[_gentau_visPt.index(maxPt)]
                maxPhi = _gentau_visPhi[_gentau_visPt.index(maxPt)]

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(maxPt, maxEta, maxPhi, 0)

                if abs(gentau.Eta()) < options.etaEr: # skip taus out of acceptance

                    denominator_ptBins_CE.Fill(gentau.Pt())

                    minatedMatched = False
                    highestMinatedL1Pt = -99.9

                    # loop over TauMinator taus
                    for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_quality):

                        l1tau = ROOT.TLorentzVector()
                        l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                        if abs(l1tau.Eta()) > options.etaEr: continue # skip taus out of acceptance
                        #if abs(l1tau.Eta()) < options.CBCEsplit:
                        #    if l1tauId < qualityCut_CB: continue
                        #else:
                        #    if l1tauId < qualityCut_CE: continue

                        # check matching
                        if gentau.DeltaR(l1tau)<0.3:
                            # keep only L1 match with highest pT
                            if l1tau.Pt()>highestMinatedL1Pt:
                                minatedMatched = True
                                highestMinatedL1Pt = l1tau.Pt()

                    # fill numerator histograms for every thresholds
                    for i, thr in enumerate(online_thresholds): 
                        if minatedMatched and highestMinatedL1Pt>float(thr):
                                minated_passing_ptBins_CE[i].Fill(gentau.Pt())


            ###########################################################################
            # OVERALL TURNON
            maxPt = -99.
            for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
                # if abs(genEta) < 1.5: continue
                if genPt > maxPt: maxPt = genPt

            # if maxPt < 0:
            #     for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
            #         if genPt > maxPt: maxPt = genPt

            if maxPt > 0:
                maxEta = _gentau_visEta[_gentau_visPt.index(maxPt)]
                maxPhi = _gentau_visPhi[_gentau_visPt.index(maxPt)]

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(maxPt, maxEta, maxPhi, 0)

                if abs(gentau.Eta()) < options.etaEr: # skip taus out of acceptance

                    denominator_ptBins.Fill(gentau.Pt())
                    if gentau.Pt() > 40: denominator_etaBins.Fill(gentau.Eta())

                    minatedMatched = False
                    highestMinatedL1Pt = -99.9

                    # loop over TauMinator taus
                    for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_quality):

                        l1tau = ROOT.TLorentzVector()
                        l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                        if abs(l1tau.Eta()) > options.etaEr: continue # skip taus out of acceptance
                        #if abs(l1tau.Eta()) < options.CBCEsplit:
                        #    if l1tauId < qualityCut_CB: continue
                        #else:
                        #    if l1tauId < qualityCut_CE: continue

                        # check matching
                        if gentau.DeltaR(l1tau)<0.3:
                            # keep only L1 match with highest pT
                            if l1tau.Pt()>highestMinatedL1Pt:
                                minatedMatched = True
                                highestMinatedL1Pt = l1tau.Pt()

                    # fill numerator histograms for every thresholds
                    for i, thr in enumerate(online_thresholds): 
                        if minatedMatched and highestMinatedL1Pt>float(thr):
                            minated_passing_ptBins[i].Fill(gentau.Pt())
                            if gentau.Pt() > 40: minated_passing_etaBins[i].Fill(gentau.Eta())


        # end of the loop over the events
        #################################

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated = []
        turnonsMinated_CB = []
        turnonsMinated_CE = []
        etaEffMinated = []

        for i, thr in enumerate(online_thresholds):
            turnonsMinated.append(ROOT.TGraphAsymmErrors(minated_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated_CB.append(ROOT.TGraphAsymmErrors(minated_passing_ptBins_CB[i], denominator_ptBins_CB, "cp"))
            turnonsMinated_CE.append(ROOT.TGraphAsymmErrors(minated_passing_ptBins_CE[i], denominator_ptBins_CE, "cp"))

            etaEffMinated.append(ROOT.TGraphAsymmErrors(minated_passing_etaBins[i], denominator_etaBins, "cp"))

        # save to file 

        fileout = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+"_CB"+WP_CB+"_CE"+WP_CE+".root","RECREATE")
        denominator_ptBins.Write()
        denominator_ptBins_CB.Write()
        denominator_ptBins_CE.Write()
        denominator_etaBins.Write()
        for i, thr in enumerate(online_thresholds): 
            minated_passing_ptBins[i].Write()
            minated_passing_ptBins_CB[i].Write()
            minated_passing_ptBins_CE[i].Write()

            minated_passing_etaBins[i].Write()

            turnonsMinated[i].Write()
            turnonsMinated_CB[i].Write()
            turnonsMinated_CE[i].Write()

            etaEffMinated[i].Write()


        fileout.Close()

    else:
        filein = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+"_CB"+WP_CB+"_CE"+WP_CE+".root","READ")

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated = []
        turnonsMinated_CB = []
        turnonsMinated_CE = []

        etaEffMinated = []

        for i, thr in enumerate(online_thresholds):
            turnonsMinated.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
            turnonsMinated_CB.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_ptBins_CB_by_denominator_ptBins_CB"))
            turnonsMinated_CE.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_ptBins_CE_by_denominator_ptBins_CE"))

            etaEffMinated.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))


    turnons_dict = {}
    turnons_dict_CB = {}
    turnons_dict_CE = {}
    etaeffs_dict = {}
    mapping_dict = {'threshold':[],
                    'CB'+WP_CB+'_CE'+WP_CE+'_pt95':[], 'CB'+WP_CB+'_CE'+WP_CE+'_pt90':[], 'CB'+WP_CB+'_CE'+WP_CE+'_pt50':[],
                    'CB'+WP_CB+'_CE'+WP_CE+'_slope_pt95' : 0.0, 'CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt95': 0.0,
                    'CB'+WP_CB+'_CE'+WP_CE+'_slope_pt90' : 0.0, 'CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt90': 0.0,
                    'CB'+WP_CB+'_CE'+WP_CE+'_slope_pt50' : 0.0, 'CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt50': 0.0}
    mapping_dict_CB = {'threshold':[],
                       'CB'+WP_CB+'_pt95':[], 'CB'+WP_CB+'_pt90':[], 'CB'+WP_CB+'_pt50':[],
                       'CB'+WP_CB+'_slope_pt95' : 0.0, 'CB'+WP_CB+'_intercept_pt95': 0.0,
                       'CB'+WP_CB+'_slope_pt90' : 0.0, 'CB'+WP_CB+'_intercept_pt90': 0.0,
                       'CB'+WP_CB+'_slope_pt50' : 0.0, 'CB'+WP_CB+'_intercept_pt50': 0.0}
    mapping_dict_CE = {'threshold':[],
                       'CE'+WP_CE+'_pt95':[], 'CE'+WP_CE+'_pt90':[], 'CE'+WP_CE+'_pt50':[],
                       'CE'+WP_CE+'_slope_pt95' : 0.0, 'CE'+WP_CE+'_intercept_pt95': 0.0,
                       'CE'+WP_CE+'_slope_pt90' : 0.0, 'CE'+WP_CE+'_intercept_pt90': 0.0,
                       'CE'+WP_CE+'_slope_pt50' : 0.0, 'CE'+WP_CE+'_intercept_pt50': 0.0}

    for i, thr in enumerate(online_thresholds):
        turnons_dict['turnonAtwpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'] = [[],[],[]]

        etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'] = [[],[],[]]

        for ibin in range(0,turnonsMinated[i].GetN()):
            turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0].append(turnonsMinated[i].GetPointY(ibin))
            turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][1].append(turnonsMinated[i].GetErrorYlow(ibin))
            turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][2].append(turnonsMinated[i].GetErrorYhigh(ibin))
            turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0].append(turnonsMinated_CB[i].GetPointY(ibin))
            turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1].append(turnonsMinated_CB[i].GetErrorYlow(ibin))
            turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2].append(turnonsMinated_CB[i].GetErrorYhigh(ibin))
            turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0].append(turnonsMinated_CE[i].GetPointY(ibin))
            turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1].append(turnonsMinated_CE[i].GetErrorYlow(ibin))
            turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2].append(turnonsMinated_CE[i].GetErrorYhigh(ibin))

        for ibin in range(0,etaEffMinated[i].GetN()):
            etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'][0].append(etaEffMinated[i].GetPointY(ibin))
            etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'][1].append(etaEffMinated[i].GetErrorYlow(ibin))
            etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'][2].append(etaEffMinated[i].GetErrorYhigh(ibin))
        
        ### ONLINE TO OFFILNE MAPPING AS MENU DOES
        # efficiency = turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0]
        # er_dn = turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][1]
        # er_up = turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][2]
        # xbins = 0.5 * np.array(ptBins[1:] + ptBins[:-1])
        # K1 = []
        # for i in range(len(efficiency)): K1.append(1 / (er_dn[i] + er_up[i]) / (er_up[i] + er_dn[i]))
        # mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt95'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.95))
        # mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.90))
        # mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt50'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.50))

        # efficiency = turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0]
        # er_dn = turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1]
        # er_up = turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2]
        # xbins = 0.5 * np.array(ptBins[1:] + ptBins[:-1])
        # K1 = []
        # for i in range(len(efficiency)): K1.append(1 / (er_dn[i] + er_up[i]) / (er_up[i] + er_dn[i]))
        # mapping_dict_CB['CB'+WP_CB+'_pt95'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.95))
        # mapping_dict_CB['CB'+WP_CB+'_pt90'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.90))
        # mapping_dict_CB['CB'+WP_CB+'_pt50'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.50))

        # efficiency = turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0]
        # er_dn = turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1]
        # er_up = turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2]
        # xbins = 0.5 * np.array(ptBins[1:] + ptBins[:-1])
        # K1 = []
        # for i in range(len(efficiency)): K1.append(1 / (er_dn[i] + er_up[i]) / (er_up[i] + er_dn[i]))
        # mapping_dict_CE['CE'+WP_CE+_pt95'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.95))
        # mapping_dict_CE['CE'+WP_CE+_pt90'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.90))
        # mapping_dict_CE['CE'+WP_CE+_pt50'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.50))


        ### ONLINE TO OFFILNE MAPPING AS I DO
        mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt95'].append(MYinterpolate(offline_pts, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], 0.95))
        mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'].append(MYinterpolate(offline_pts, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], 0.90))
        mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt50'].append(MYinterpolate(offline_pts, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], 0.50))

        mapping_dict_CB['CB'+WP_CB+'_pt95'].append(MYinterpolate(offline_pts, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], 0.95))
        mapping_dict_CB['CB'+WP_CB+'_pt90'].append(MYinterpolate(offline_pts, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], 0.90))
        mapping_dict_CB['CB'+WP_CB+'_pt50'].append(MYinterpolate(offline_pts, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], 0.50))

        mapping_dict_CE['CE'+WP_CE+'_pt95'].append(MYinterpolate(offline_pts, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], 0.95))
        mapping_dict_CE['CE'+WP_CE+'_pt90'].append(MYinterpolate(offline_pts, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], 0.90))
        mapping_dict_CE['CE'+WP_CE+'_pt50'].append(MYinterpolate(offline_pts, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], 0.50))


    ##################################################################################
    # FIT SCALINGS

    def line(x, A, B):
        return A * x + B

    p0 = [1,1]
    popt,    pcov = curve_fit(line, online_thresholds[25:80], mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt95'][25:80], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:80], mapping_dict_CB['CB'+WP_CB+'_pt95'][25:80], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60], mapping_dict_CE['CE'+WP_CE+'_pt95'][25:60], p0, maxfev=5000)
    mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_slope_pt95'] = popt[0]
    mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt95'] = popt[1]
    mapping_dict_CB['CB'+WP_CB+'_slope_pt95'] = popt_CB[0]
    mapping_dict_CB['CB'+WP_CB+'_intercept_pt95'] = popt_CB[1]
    mapping_dict_CE['CE'+WP_CE+'_slope_pt95'] = popt_CE[0]
    mapping_dict_CE['CE'+WP_CE+'_intercept_pt95'] = popt_CE[1]
    print('\n95% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    popt,    pcov = curve_fit(line, online_thresholds[25:80], mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'][25:80], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:80], mapping_dict_CB['CB'+WP_CB+'_pt90'][25:80], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60], mapping_dict_CE['CE'+WP_CE+'_pt90'][25:60], p0, maxfev=5000)
    # mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_slope_pt90'] = popt[0]
    # mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt90'] = popt[1]
    # mapping_dict_CB['CB'+WP_CB+'_slope_pt90'] = popt_CB[0]
    # mapping_dict_CB['CB'+WP_CB+'_intercept_pt90'] = popt_CB[1]
    # mapping_dict_CE['CE'+WP_CE+'_slope_pt90'] = popt_CE[0]
    # mapping_dict_CE['CE'+WP_CE+'_intercept_pt90'] = popt_CE[1]
    print('\n90% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    popt,    pcov = curve_fit(line, np.array(online_thresholds)[[25,30,40,50,60,70,80]], np.array(mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'])[[25,30,40,50,60,70,80]], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, np.array(online_thresholds)[[25,30,40,50,60,70,80]], np.array(mapping_dict_CB['CB'+WP_CB+'_pt90'])[[25,30,40,50,60,70,80]], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, np.array(online_thresholds)[[25,30,40,50,60,70,80]], np.array(mapping_dict_CE['CE'+WP_CE+'_pt90'])[[25,30,40,50,60,70,80]], p0, maxfev=5000)
    mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_slope_pt90'] = popt[0]
    mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt90'] = popt[1]
    mapping_dict_CB['CB'+WP_CB+'_slope_pt90'] = popt_CB[0]
    mapping_dict_CB['CB'+WP_CB+'_intercept_pt90'] = popt_CB[1]
    mapping_dict_CE['CE'+WP_CE+'_slope_pt90'] = popt_CE[0]
    mapping_dict_CE['CE'+WP_CE+'_intercept_pt90'] = popt_CE[1]
    print('\n90% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    plt.figure(figsize=(10,10))
    plt.plot(np.array(online_thresholds)[[25,30,40,50,60,70,80]], np.array(mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'])[[25,30,40,50,60,70,80]], ls='None', label='Overall', marker='o', color='black')
    plt.plot(np.array(online_thresholds)[[25,30,40,50,60,70,80]], np.array(mapping_dict_CB['CB'+WP_CB+'_pt90'])[[25,30,40,50,60,70,80]], ls='None', label='Barrel - WP '+WP_CB, marker='o', color='blue')
    plt.plot(np.array(online_thresholds)[[25,30,40,50,60,70,80]], np.array(mapping_dict_CE['CE'+WP_CE+'_pt90'])[[25,30,40,50,60,70,80]], ls='None', label='Endcap - WP '+WP_CE, marker='o', color='orange')
    plt.plot(np.arange(0,200,1), line(np.arange(0,200,1), *popt),    ls='--', lw=2, color='black')
    plt.plot(np.arange(0,200,1), line(np.arange(0,200,1), *popt_CB), ls='--', lw=2, color='blue')
    plt.plot(np.arange(0,200,1), line(np.arange(0,200,1), *popt_CE), ls='--', lw=2, color='orange')
    plt.legend(loc = 'upper left', fontsize=14)
    plt.ylim(0., 150)
    plt.xlim(0., 100)
    plt.xlabel(r'$p_{T}^{L1,\tau}$')
    plt.ylabel(r'$p_{T}^{Offline, \tau}$ at 90% efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/scalingFit_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
    plt.close()

    popt,    pcov = curve_fit(line, online_thresholds[25:80], mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt50'][25:80], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:80], mapping_dict_CB['CB'+WP_CB+'_pt50'][25:80], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60], mapping_dict_CE['CE'+WP_CE+'_pt50'][25:60], p0, maxfev=5000)
    mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_slope_pt50'] = popt[0]
    mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_intercept_pt50'] = popt[1]
    mapping_dict_CB['CB'+WP_CB+'_slope_pt50'] = popt_CB[0]
    mapping_dict_CB['CB'+WP_CB+'_intercept_pt50'] = popt_CB[1]
    mapping_dict_CE['CE'+WP_CE+'_slope_pt50'] = popt_CE[0]
    mapping_dict_CE['CE'+WP_CE+'_intercept_pt50'] = popt_CE[1]
    print('\n50% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    save_obj(mapping_dict, perfdir+'/turnons'+tag+'/online2offline_mapping_CB'+WP_CB+'_CE'+WP_CE+'.pkl')
    save_obj(mapping_dict_CB, perfdir+'/turnons'+tag+'/online2offline_mapping_CB'+WP_CB+'.pkl')
    save_obj(mapping_dict_CE, perfdir+'/turnons'+tag+'/online2offline_mapping_CE'+WP_CE+'.pkl')


    ##################################################################################
    # PLOT TURNONS

    # x_Emyr_CB = [9.032258064516128, 15.11921458625526, 21.00981767180925, 27.09677419354839, 32.987377279102375, 38.87798036465638, 45.16129032258064, 51.051893408134646, 56.94249649368864, 63.02945301542778, 68.92005610098177, 75.0070126227209, 80.8976157082749, 86.98457223001401, 93.07152875175315, 98.96213183730715, 105.24544179523141, 110.93969144460027, 116.83029453015428, 123.11360448807852, 129.00420757363253, 135.09116409537165, 140.98176718092566, 146.87237026647966]
    # y_Emyr_CB = [0.04236006051437213, 0.24810892586989397, 0.6066565809379727, 0.7715582450832071, 0.8865355521936459, 0.9273827534039334, 0.9531013615733736, 0.956127080181543, 0.966717095310136, 0.966717095310136, 0.9697428139183055, 0.9621785173978818, 0.9757942511346444, 0.9803328290468986, 0.9818456883509833, 0.9788199697428138, 0.9848714069591527, 0.9818456883509833, 0.9833585476550679, 0.9818456883509833, 0.9848714069591527, 0.9773071104387291, 0.9712556732223903, 0.9863842662632374]

    # x_Emyr_CE = [2.9453015427769955, 8.835904628330997, 14.922861150070126, 21.00981767180925, 27.09677419354839, 32.987377279102375, 38.87798036465638, 44.964936886395506, 51.051893408134646, 56.94249649368864, 63.02945301542778, 68.92005610098177, 75.0070126227209, 80.8976157082749, 86.78821879382889, 92.87517531556801, 98.96213183730715, 105.04908835904628, 110.93969144460027, 116.83029453015428, 122.9172510518934, 128.8078541374474, 134.89481065918653, 140.98176718092566, 146.87237026647966]
    # y_Emyr_CE = [0.021180030257186067, 0.04689863842662634, 0.25113464447806344, 0.475037821482602, 0.6369137670196671, 0.7473524962178517, 0.8033282904689862, 0.8184568835098335, 0.8668683812405445, 0.8835098335854765, 0.8865355521936459, 0.8835098335854765, 0.8940998487140694, 0.9077155824508321, 0.9213313161875945, 0.9137670196671708, 0.9258698940998487, 0.9213313161875945, 0.9425113464447805, 0.8850226928895611, 0.939485627836611, 0.9198184568835097, 0.9273827534039334, 0.9697428139183055, 0.8577912254160363]

    # i = 0
    # plt.figure(figsize=(10,10))
    # thr=0
    # plt.errorbar(offline_pts,turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'CB - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='blue', alpha=0.75)
    # plt.errorbar(offline_pts,turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'CE - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='orange', alpha=0.75)
    # plt.errorbar(x_Emyr_CB,y_Emyr_CB, xerr=1, yerr=0.01, ls='--', label=r'CB Menu - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='blue')
    # plt.errorbar(x_Emyr_CE,y_Emyr_CE, xerr=1, yerr=0.01, ls='--', label=r'CE Menu - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='orange')
    # plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # # plt.xscale('log')
    # plt.xlabel(r'$p_{T}^{Gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/turnons'+tag+'/turnons_Emyr_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
    # plt.close()


    cmap = matplotlib.cm.get_cmap('tab20c'); i=0
    
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            # p0 = [1, thr, 1] 
            # popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            # plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{Gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
    plt.close()

    ##################################################################################
    # PLOT TURNONS CB
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            # p0 = [1, thr, 1] 
            # popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            # plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            # spline = interpolate.Akima1DInterpolator(offline_pts, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0])
            # xs = np.arange(0,200,1)
            # plt.plot(xs, spline(xs), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{Gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_CB'+WP_CB+'.pdf')
    plt.close()

    ##################################################################################
    # PLOT TURNONS CE
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            # p0 = [1, thr, 1] 
            # popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            # plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            # spline = interpolate.Akima1DInterpolator(offline_pts, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0])
            # xs = np.arange(0,200,1)
            # plt.plot(xs, spline(xs), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{Gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_CE'+WP_CE+'.pdf')
    plt.close()


    ##################################################################################
    # PLOT EFFICIENCIES VS ETA
    i = 0
    plt.figure(figsize=(10,10))
    plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAtwpAt0GeV'][0],yerr=[etaeffs_dict['efficiencyVsEtaAtwpAt0GeV'][1], etaeffs_dict['efficiencyVsEtaAtwpAt0GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (32), lw=2, marker='o', color=cmap(i))
    plt.hlines(0.90, -3.0, 3.0, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, -3.0, 3.0, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(-3.0, 3.0)
    plt.xlabel(r'$\eta^{Gen,\tau}$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/efficiencyVSeta_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
    plt.close()

    ##################################################################################
    # PLOT ONLINE TO OFFLINE MAPPING
    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(20, 100)
    plt.ylim(20, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
    plt.close()

    p0 = [1,1]
    popt, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'][25:100], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['CB'+WP_CB+'_pt90'][25:100], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60], mapping_dict_CE['CE'+WP_CE+'_pt90'][25:60], p0, maxfev=5000)

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt90'], label='Overall', linewidth=2, color='black')
    plt.plot(online_thresholds, mapping_dict_CB['CB'+WP_CB+'_pt90'], label='Barrel', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict_CE['CE'+WP_CE+'_pt90'], label='Endcap', linewidth=2, color='orange')
    plt.plot(online_thresholds, line(online_thresholds, *popt), label='Fit Overall', linewidth=2, ls='--', color='black')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CB), label='Fit Barrel', linewidth=2, ls='--', color='blue')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CE), label='Fit Endcap', linewidth=2, ls='--', color='orange')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(25, 90)
    plt.ylim(25, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_CB'+WP_CB+'_CE'+WP_CE+'_pt90_split.pdf')
    plt.close()

    p0 = [1,1]
    popt, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt50'][25:100], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['CB'+WP_CB+'_pt50'][25:100], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CE['CE'+WP_CE+'_pt50'][25:100], p0, maxfev=5000)

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['CB'+WP_CB+'_CE'+WP_CE+'_pt50'], label='Overall', linewidth=2, color='black')
    plt.plot(online_thresholds, mapping_dict_CB['CB'+WP_CB+'_pt50'], label='Barrel', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict_CE['CE'+WP_CE+'_pt50'], label='Endcap', linewidth=2, color='orange')
    plt.plot(online_thresholds, line(online_thresholds, *popt), label='Fit Overall', linewidth=2, ls='--', color='black')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CB), label='Fit Barrel', linewidth=2, ls='--', color='blue')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CE), label='Fit Endcap', linewidth=2, ls='--', color='orange')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(25, 90)
    plt.ylim(25, 100)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_CB'+WP_CB+'_CE'+WP_CE+'_pt50_split.pdf')
    plt.close()

    ###################################################################################
    # DP NOTE PLOTS

    # APPROXIMATE ERROR FUNCTION
    def ApproxErf(arg):
        erflim = 5.0;
        if arg > erflim:
            return 1.0
        if arg < -erflim:
            return -1.0

        return sp.erf(arg)

    # CUMULATIVE CRYSTAL-BALL : this is the most important part of the turnon and can be used for both eg and tau
    def CB(x, mean=1, sigma=1, alpha=1, n=1, norm=1):
        pi = np.pi
        sqrt2 = np.sqrt(2)
        sqrtPiOver2 = np.sqrt(np.pi / 2)

        # Variable std deviation
        sig = abs(sigma)
        t = (x - mean)/sig
        if alpha < 0:
            t = -t

        # Crystal Ball part
        absAlpha = abs(alpha)
        A = pow(n / absAlpha, n) * np.exp(-0.5 * absAlpha * absAlpha)
        B = absAlpha - n / absAlpha
        C = n / absAlpha * np.exp(-0.5*absAlpha*absAlpha) / (n - 1)
        D = (1 + ApproxErf(absAlpha / sqrt2)) * sqrtPiOver2
        N = norm / (D + C)

        if t <= absAlpha:
            crystalBall = N * (1 + ApproxErf( t / sqrt2 )) * sqrtPiOver2
        else:
            crystalBall = N * (D +  A * (1/pow(t-B,n-1) - 1/pow(absAlpha - B,n-1)) / (1 - n))

        return crystalBall
    vectCB = np.vectorize(CB)

    # ARCTAN TRUNCATED IN THE LOW TAIL : used with the CB CDF fits well and almost out of the box the tau turnons
    def ApproxATAN(x, xturn=1, p=1, width=1):
        pi = np.pi

        # Arctan part
        arctan = 0.
        if x < xturn:
            arctan = p
        if x >= xturn:
            arctan = pow(ApproxErf((x - xturn) / 5.), 2) * 2. * (1. - p) / pi * np.arctan(pi / 80. * width * (x - xturn)) + p

        return arctan
    vectApproxATAN = np.vectorize(ApproxATAN)

    def CBconvATAN(x, mean=1, sigma=1, alpha=1, n=1, norm=1, xturn=1, p=1, width=1):
        return convolve(vectCB(x, mean=mean, sigma=sigma, alpha=alpha, n=n, norm=norm), vectApproxATAN(x, xturn=xturn, p=p, width=width), mode='full', method='direct')
    vectCBconvATAN = np.vectorize(CBconvATAN)


    WP_CB_text = ""
    WP_CE_text = ""
    if WP_CB == "90": WP_CB_text = 'Tight'
    if WP_CE == "90": WP_CE_text = 'Tight'
    if WP_CB == "95": WP_CB_text = 'Medium'
    if WP_CE == "95": WP_CE_text = 'Medium'

    plt.figure(figsize=(12,12))
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', ls='--', zorder=0)
    plt.hlines(1.00, 0, 2000, lw=2, color='dimgray', ls='--', zorder=0)
    plt.errorbar(offline_pts,turnons_dict['turnonAtwpAt0GeV'][0],xerr=2.5,yerr=[turnons_dict['turnonAtwpAt0GeV'][1], turnons_dict['turnonAtwpAt0GeV'][2]], ls='None', label=r'TauMinator ('+WP_CB_text+'-'+WP_CE_text+' WP)', marker='o', markersize='10', lw=3, color='black', zorder=3)
    plt.errorbar(offline_pts,turnons_dict_CB['turnonAtwpAt0GeV'][0],xerr=2.5,yerr=[turnons_dict_CB['turnonAtwpAt0GeV'][1], turnons_dict_CB['turnonAtwpAt0GeV'][2]], ls='None', label=r'TauMinator ('+WP_CB_text+' WP) - Barrel', marker='s', markersize='10', lw=3, color='blue', zorder=1)
    plt.errorbar(offline_pts,turnons_dict_CE['turnonAtwpAt0GeV'][0],xerr=2.5,yerr=[turnons_dict_CE['turnonAtwpAt0GeV'][1], turnons_dict_CE['turnonAtwpAt0GeV'][2]], ls='None', label=r'TauMinator ('+WP_CE_text+' WP) - Endcap', marker='^', markersize='10', lw=3, color='darkorange', zorder=2)

    p0 =          [10    ,    3.,   3. , 100.,     0.95,   10., 0.8,   10.]
    param_bounds=([10-10.,    1.,   0.1,   1.,     0.9 ,    0., 0.2,    1.],
                  [10+10.,   10.,  10. , 200.,     1.  ,  110., 1. ,  100.])
    popt, pcov = curve_fit(vectCBconvATAN, offline_pts, turnons_dict['turnonAtwpAt0GeV'][0], p0, maxfev=5000, bounds=param_bounds)
    plt.plot(offline_pts, vectCBconvATAN(offline_pts, *popt), '-', label='_', lw=3, color='black', zorder=3)

    popt, pcov = curve_fit(vectCBconvATAN, offline_pts, turnons_dict_CB['turnonAtwpAt0GeV'][0], p0, maxfev=5000, bounds=param_bounds)
    plt.plot(offline_pts, vectCBconvATAN(offline_pts, *popt), '-', label='_', lw=3, color='blue', zorder=1)

    popt, pcov = curve_fit(vectCBconvATAN, offline_pts, turnons_dict_CE['turnonAtwpAt0GeV'][0], p0, maxfev=5000, bounds=param_bounds)
    plt.plot(offline_pts, vectCBconvATAN(offline_pts, *popt), '-', label='_', lw=3, color='darkorange', zorder=2)

    plt.legend(loc = 'lower right', fontsize=20)
    plt.ylim(0., 1.05)
    plt.xlim(0., 140.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{Gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation Preliminary ', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/DP_turnons_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
    plt.close()




    fileinTight = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+"_CB90_CE90.root","READ")
    fileinMedium = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+"_CB95_CE95.root","READ")
    fileinLoose = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+"_CB99_CE99.root","READ")
    
    etaeffsTight = fileinTight.Get("divide_minated_passing_thr0_etaBins_by_denominator_etaBins")
    etaeffsMedium = fileinMedium.Get("divide_minated_passing_thr0_etaBins_by_denominator_etaBins")
    etaeffsLoose = fileinLoose.Get("divide_minated_passing_thr0_etaBins_by_denominator_etaBins")
    
    tunroneffsTight = fileinTight.Get("divide_minated_passing_thr0_ptBins_by_denominator_ptBins")
    tunroneffsMedium = fileinMedium.Get("divide_minated_passing_thr0_ptBins_by_denominator_ptBins")
    tunroneffsLoose = fileinLoose.Get("divide_minated_passing_thr0_ptBins_by_denominator_ptBins")

    etaeffs_dict = {}
    etaeffs_dict['Tight'] = [[],[],[]]
    etaeffs_dict['Medium'] = [[],[],[]]
    etaeffs_dict['Loose'] = [[],[],[]]

    turnoneffs_dict = {}
    turnoneffs_dict['Tight'] = [[],[],[]]
    turnoneffs_dict['Medium'] = [[],[],[]]
    turnoneffs_dict['Loose'] = [[],[],[]]

    for ibin in range(0,etaeffsTight.GetN()):
        etaeffs_dict['Tight'][0].append(etaeffsTight.GetPointY(ibin))
        etaeffs_dict['Tight'][1].append(etaeffsTight.GetErrorYlow(ibin))
        etaeffs_dict['Tight'][2].append(etaeffsTight.GetErrorYhigh(ibin))

        etaeffs_dict['Medium'][0].append(etaeffsMedium.GetPointY(ibin))
        etaeffs_dict['Medium'][1].append(etaeffsMedium.GetErrorYlow(ibin))
        etaeffs_dict['Medium'][2].append(etaeffsMedium.GetErrorYhigh(ibin))

        etaeffs_dict['Loose'][0].append(etaeffsLoose.GetPointY(ibin))
        etaeffs_dict['Loose'][1].append(etaeffsLoose.GetErrorYlow(ibin))
        etaeffs_dict['Loose'][2].append(etaeffsLoose.GetErrorYhigh(ibin))

    for ibin in range(0,tunroneffsTight.GetN()):
        turnoneffs_dict['Tight'][0].append(tunroneffsTight.GetPointY(ibin))
        turnoneffs_dict['Tight'][1].append(tunroneffsTight.GetErrorYlow(ibin))
        turnoneffs_dict['Tight'][2].append(tunroneffsTight.GetErrorYhigh(ibin))

        turnoneffs_dict['Medium'][0].append(tunroneffsMedium.GetPointY(ibin))
        turnoneffs_dict['Medium'][1].append(tunroneffsMedium.GetErrorYlow(ibin))
        turnoneffs_dict['Medium'][2].append(tunroneffsMedium.GetErrorYhigh(ibin))

        turnoneffs_dict['Loose'][0].append(tunroneffsLoose.GetPointY(ibin))
        turnoneffs_dict['Loose'][1].append(tunroneffsLoose.GetErrorYlow(ibin))
        turnoneffs_dict['Loose'][2].append(tunroneffsLoose.GetErrorYhigh(ibin))

    plt.figure(figsize=(12,12))
    plt.hlines(0.90, -3, 3, lw=2, color='dimgray', ls='--', zorder=0)
    plt.hlines(1.00, -3, 3, lw=2, color='dimgray', ls='--', zorder=0)
    plt.errorbar(eta_bins_centers,etaeffs_dict['Tight'][0],xerr=0.1,yerr=[etaeffs_dict['Tight'][1], etaeffs_dict['Tight'][2]], ls='None', label=r'TauMinator (Tight-Tight WP)', lw=3, marker='o', markersize='10', color="#6996e3", zorder=0)
    plt.errorbar(eta_bins_centers,etaeffs_dict['Medium'][0],xerr=0.1,yerr=[etaeffs_dict['Medium'][1], etaeffs_dict['Medium'][2]], ls='None', label=r'TauMinator (Medium-Medium WP)', lw=3, marker='s', markersize='10', color="#4060c8", zorder=1)
    plt.errorbar(eta_bins_centers,etaeffs_dict['Loose'][0],xerr=0.1,yerr=[etaeffs_dict['Loose'][1], etaeffs_dict['Loose'][2]], ls='None', label=r'TauMinator (Loose-Loose WP)', lw=3, marker='^', markersize='10', color="#1a318b", zorder=2)
    leg = plt.legend(loc = 'lower right', title=r'$p_{T}^{Gen, \tau}>40GeV$', title_fontsize=20, fontsize=20)
    leg._legend_box.align = "left"
    plt.ylim(0.5, 1.05)
    plt.xlim(-3.0, 3.0)
    plt.xlabel(r'$\eta^{Gen,\tau}$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/DP_efficiencyVSeta.pdf')
    plt.close()



    plt.figure(figsize=(12,12))
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', ls='--', zorder=0)
    plt.hlines(1.00, 0, 2000, lw=2, color='dimgray', ls='--', zorder=0)
    plt.errorbar(offline_pts,turnoneffs_dict['Tight'][0],xerr=2.5,yerr=[turnoneffs_dict['Tight'][1], turnoneffs_dict['Tight'][2]], ls='None', label=r'TauMinator (Tight-Tight WP)', marker='o', markersize='10', lw=3, color='#6996e3', zorder=3)
    plt.errorbar(offline_pts,turnoneffs_dict['Medium'][0],xerr=2.5,yerr=[turnoneffs_dict['Medium'][1], turnoneffs_dict['Medium'][2]], ls='None', label=r'TauMinator (Medium-Medium WP)', marker='s', markersize='10', lw=3, color='#4060c8', zorder=1)
    plt.errorbar(offline_pts,turnoneffs_dict['Loose'][0],xerr=2.5,yerr=[turnoneffs_dict['Loose'][1], turnoneffs_dict['Loose'][2]], ls='None', label=r'TauMinator (Loose-Loose WP)', marker='^', markersize='10', lw=3, color='#1a318b', zorder=2)

    p0 =          [10    ,    3.,   3. , 100.,     0.95,   10., 0.8,   10.]
    param_bounds=([10-10.,    1.,   0.1,   1.,     0.9 ,    0., 0.2,    1.],
                  [10+10.,   10.,  10. , 200.,     1.  ,  110., 1. ,  100.])
    popt, pcov = curve_fit(vectCBconvATAN, offline_pts, turnoneffs_dict['Tight'][0], p0, maxfev=5000, bounds=param_bounds)
    plt.plot(offline_pts, vectCBconvATAN(offline_pts, *popt), '-', label='_', lw=3, color='#6996e3', zorder=3)

    popt, pcov = curve_fit(vectCBconvATAN, offline_pts, turnoneffs_dict['Medium'][0], p0, maxfev=5000, bounds=param_bounds)
    plt.plot(offline_pts, vectCBconvATAN(offline_pts, *popt), '-', label='_', lw=3, color='#4060c8', zorder=1)

    popt, pcov = curve_fit(vectCBconvATAN, offline_pts, turnoneffs_dict['Loose'][0], p0, maxfev=5000, bounds=param_bounds)
    plt.plot(offline_pts, vectCBconvATAN(offline_pts, *popt), '-', label='_', lw=3, color='#1a318b', zorder=2)

    plt.legend(loc = 'lower right', fontsize=20)
    plt.ylim(0., 1.05)
    plt.xlim(0., 140.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{Gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation Preliminary ', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/DP_turnons.pdf')
    plt.close()
