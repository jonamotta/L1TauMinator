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
    parser.add_option("--NtupleV",          dest="NtupleV",        default=None)
    parser.add_option("--v",                dest="v",              default=None)
    parser.add_option("--date",             dest="date",           default=None)
    parser.add_option('--loop',             dest='loop',           default=False, action="store_true")
    parser.add_option('--etaEr',            dest='etaEr',          default=3.0,   type=float)
    parser.add_option('--caloClNxM',        dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",        dest="seedEtCut",      default="2p5")
    parser.add_option("--WP",               dest="WP",             default="99")
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/rate'+tag)

    WP = options.WP

    if options.loop:
        #passing histograms (numerators)
        singleTau_onlinePtProgression = ROOT.TH1F("singleTau_onlinePtProgression","singleTau_onlinePtProgression",240,0.,240.)
        singleTau_offline95PtProgression = ROOT.TH1F("singleTau_offline95PtProgression","singleTau_offline95PtProgression",240,0.,240.)
        singleTau_offline90PtProgression = ROOT.TH1F("singleTau_offline90PtProgression","singleTau_offline90PtProgression",240,0.,240.)
        singleTau_offline50PtProgression = ROOT.TH1F("singleTau_offline50PtProgression","singleTau_offline50PtProgression",240,0.,240.)

        diTau_onlinePtProgression = ROOT.TH2F("diTau_onlinePtProgression","diTau_onlinePtProgression",240,0.,240.,240,0.,240.)
        diTau_offline95PtProgression = ROOT.TH2F("diTau_offline95PtProgression","diTau_offline95PtProgression",240,0.,240.,240,0.,240.)
        diTau_offline90PtProgression = ROOT.TH2F("diTau_offline90PtProgression","diTau_offline90PtProgression",240,0.,240.,240,0.,240.)
        diTau_offline50PtProgression = ROOT.TH2F("diTau_offline50PtProgression","diTau_offline50PtProgression",240,0.,240.,240,0.,240.)

        singleTau_onlineRate = ROOT.TH1F("singleTau_onlineRate","singleTau_onlineRate",240,0.,240.)
        singleTau_offline95Rate = ROOT.TH1F("singleTau_offline95Rate","singleTau_offline95Rate",240,0.,240.)
        singleTau_offline90Rate = ROOT.TH1F("singleTau_offline90Rate","singleTau_offline90Rate",240,0.,240.)
        singleTau_offline50Rate = ROOT.TH1F("singleTau_offline50Rate","singleTau_offline50Rate",240,0.,240.)

        diTau_onlineRate = ROOT.TH1F("diTau_onlineRate","diTau_onlineRate",240,0.,240.)
        diTau_offline95Rate = ROOT.TH1F("diTau_offline95Rate","diTau_offline95Rate",240,0.,240.)
        diTau_offline90Rate = ROOT.TH1F("diTau_offline90Rate","diTau_offline90Rate",240,0.,240.)
        diTau_offline50Rate = ROOT.TH1F("diTau_offline50Rate","diTau_offline50Rate",240,0.,240.)

        #denominator
        denominator = 0.

        # working points
        idWp_CB = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/TauMinator_CB_ident_plots/CLTW_TauIdentifier_WPs.pkl')
        idWp_CE = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/TauMinator_CE_ident_plots/CLTW_TauIdentifier_WPs.pkl')
        mapping_dict = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping.pkl')
        online_thresholds = range(20, 175, 1)

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/MinBias_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
        inChain.Add(directory+'/Ntuple_*.root');
        nEntries = inChain.GetEntries()
        for evt in range(0, nEntries):
            if evt%1000==0: print('--> ',evt)
            # if evt == 500000: break

            entry = inChain.GetEntry(evt)

            denominator += 1.

            _l1tau_pt = list(inChain.minatedl1tau_pt)
            _l1tau_eta = list(inChain.minatedl1tau_eta)
            _l1tau_phi = list(inChain.minatedl1tau_phi)
            _l1tau_IDscore = list(inChain.minatedl1tau_IDscore)

            # filledProgression  = False
            PtObjsProgression = array('f',[-1,-1])
            IndexObjsProgression = array('i',[-1,-1])
            iOBJ = 0

            highestMinatedL1Pt = -99.9

            # loop over TauMinator taus
            for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_IDscore):
                if abs(l1tauEta) > options.etaEr: continue # skip taus out of acceptance

                # skip the absurd cases with pT->infinity
                if l1tauPt > 5000: continue
                
                if abs(l1tauEta) < 1.5:
                    if l1tauId < idWp_CB['wp'+WP]: continue
                else:
                    if l1tauId < idWp_CE['wp'+WP]: continue

                l1tau = ROOT.TLorentzVector()
                l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                # offline95tauPt = np.interp(l1tauPt, online_thresholds, mapping_dict['wp'+WP+'_pt95'])
                offline90tauPt = np.interp(l1tauPt, online_thresholds, mapping_dict['wp'+WP+'_pt90'])
                offline50tauPt = np.interp(l1tauPt, online_thresholds, mapping_dict['wp'+WP+'_pt50'])

                # single
                if l1tau.Pt()>highestMinatedL1Pt:
                    highestMinatedL1Pt = l1tau.Pt()

                # di
                if l1tau.Pt()>=PtObjsProgression[0]:
                    # tmp = ROOT.TLorentzVector()
                    # tmp.SetPtEtaPhiM(_minatedl1tau_pt[IndexObjsProgression[0]], _minatedl1tau_eta[IndexObjsProgression[0]], _minatedl1tau_phi[IndexObjsProgression[0]], 0)
                    # if l1tau.DeltaR(tmp)>0.5:
                    IndexObjsProgression[1]=IndexObjsProgression[0]
                    IndexObjsProgression[0]=iOBJ
                    PtObjsProgression[0] = l1tau.Pt()
                
                elif l1tau.Pt()>=PtObjsProgression[1]:
                    # tmp = ROOT.TLorentzVector()
                    # tmp.SetPtEtaPhiM(_minatedl1tau_pt[IndexObjsProgression[1]], _minatedl1tau_eta[IndexObjsProgression[1]], _minatedl1tau_phi[IndexObjsProgression[1]], 0)
                    # if l1tau.DeltaR(tmp)>0.5:
                    IndexObjsProgression[1]=iOBJ
                    PtObjsProgression[1] = l1tau.Pt()

                iOBJ += 1

            # single
            singleTau_onlinePtProgression.Fill(highestMinatedL1Pt)
            # singleTau_offline95PtProgression.Fill(np.interp(highestMinatedL1Pt, online_thresholds, mapping_dict['wp'+WP+'_pt95']))
            singleTau_offline90PtProgression.Fill(np.interp(highestMinatedL1Pt, online_thresholds, mapping_dict['wp'+WP+'_pt90']))
            singleTau_offline50PtProgression.Fill(np.interp(highestMinatedL1Pt, online_thresholds, mapping_dict['wp'+WP+'_pt50']))

            # di
            if IndexObjsProgression[0]>=0 and IndexObjsProgression[1]>=0:
                pt0 = PtObjsProgression[0]
                pt1 = PtObjsProgression[1]

                diTau_onlinePtProgression.Fill(pt0,pt1)
                # diTau_offline95PtProgression.Fill( np.interp(pt0, online_thresholds, mapping_dict['wp'+WP+'_pt95']) , np.interp(pt1, online_thresholds, mapping_dict['wp'+WP+'_pt95']) )
                diTau_offline90PtProgression.Fill( np.interp(pt0, online_thresholds, mapping_dict['wp'+WP+'_pt90']) , np.interp(pt1, online_thresholds, mapping_dict['wp'+WP+'_pt90']) )
                diTau_offline50PtProgression.Fill( np.interp(pt0, online_thresholds, mapping_dict['wp'+WP+'_pt50']) , np.interp(pt1, online_thresholds, mapping_dict['wp'+WP+'_pt50']) )

        # end of the loop over the events
        #################################

        scale=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf

        for i in range(0,241):
            singleTau_onlineRate.SetBinContent(i+1,singleTau_onlinePtProgression.Integral(i+1,241)/denominator*scale);
            singleTau_offline95Rate.SetBinContent(i+1,singleTau_offline95PtProgression.Integral(i+1,241)/denominator*scale);
            singleTau_offline90Rate.SetBinContent(i+1,singleTau_offline90PtProgression.Integral(i+1,241)/denominator*scale);
            singleTau_offline50Rate.SetBinContent(i+1,singleTau_offline50PtProgression.Integral(i+1,241)/denominator*scale);
            
            diTau_onlineRate.SetBinContent(i+1,diTau_onlinePtProgression.Integral(i+1,241,i+1,241)/denominator*scale);
            diTau_offline95Rate.SetBinContent(i+1,diTau_offline95PtProgression.Integral(i+1,241,i+1,241)/denominator*scale);
            diTau_offline90Rate.SetBinContent(i+1,diTau_offline90PtProgression.Integral(i+1,241,i+1,241)/denominator*scale);
            diTau_offline50Rate.SetBinContent(i+1,diTau_offline50PtProgression.Integral(i+1,241,i+1,241)/denominator*scale);

        # save to file 
        fileout = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_wp"+WP+".root","RECREATE")
        singleTau_onlinePtProgression.Write()
        singleTau_offline95PtProgression.Write()
        singleTau_offline90PtProgression.Write()
        singleTau_offline50PtProgression.Write()
        diTau_onlinePtProgression.Write()
        diTau_offline95PtProgression.Write()
        diTau_offline90PtProgression.Write()
        diTau_offline50PtProgression.Write()

        singleTau_onlineRate.Write()
        singleTau_offline95Rate.Write()
        singleTau_offline90Rate.Write()
        singleTau_offline50Rate.Write()
        diTau_onlineRate.Write()
        diTau_offline95Rate.Write()
        diTau_offline90Rate.Write()
        diTau_offline50Rate.Write()

        fileout.Close()

    else:
        filein = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_wp"+WP+".root","READ")

        # singleTau_onlinePtProgression = filein.Get("singleTau_onlinePtProgression")
        # singleTau_offline95PtProgression = filein.Get("singleTau_offline95PtProgression")
        # singleTau_offline90PtProgression = filein.Get("singleTau_offline90PtProgression")
        # singleTau_offline50PtProgression = filein.Get("singleTau_offline50PtProgression")
        # diTau_onlinePtProgression = filein.Get("diTau_onlinePtProgression")
        # diTau_offline95PtProgression = filein.Get("diTau_offline95PtProgression")
        # diTau_offline90PtProgression = filein.Get("diTau_offline90PtProgression")
        # diTau_offline50PtProgression = filein.Get("diTau_offline50PtProgression")
        singleTau_onlineRate = filein.Get("singleTau_onlineRate")
        singleTau_offline95Rate = filein.Get("singleTau_offline95Rate")
        singleTau_offline90Rate = filein.Get("singleTau_offline90Rate")
        singleTau_offline50Rate = filein.Get("singleTau_offline50Rate")
        diTau_onlineRate = filein.Get("diTau_onlineRate")
        diTau_offline95Rate = filein.Get("diTau_offline95Rate")
        diTau_offline90Rate = filein.Get("diTau_offline90Rate")
        diTau_offline50Rate = filein.Get("diTau_offline50Rate")


Xs_list = []
singleTau_onlineRate_list = []
singleTau_offline95Rate_list = []
singleTau_offline90Rate_list = []
singleTau_offline50Rate_list = []
diTau_onlineRate_list = []
diTau_offline95Rate_list = []
diTau_offline90Rate_list = []
diTau_offline50Rate_list = []
for ibin in range(1,singleTau_onlineRate.GetNbinsX()+1):
    Xs_list.append(ibin)
    singleTau_onlineRate_list.append(singleTau_onlineRate.GetBinContent(ibin))
    singleTau_offline95Rate_list.append(singleTau_offline95Rate.GetBinContent(ibin))
    singleTau_offline90Rate_list.append(singleTau_offline90Rate.GetBinContent(ibin))
    singleTau_offline50Rate_list.append(singleTau_offline50Rate.GetBinContent(ibin))
    diTau_onlineRate_list.append(diTau_onlineRate.GetBinContent(ibin))
    diTau_offline95Rate_list.append(diTau_offline95Rate.GetBinContent(ibin))
    diTau_offline90Rate_list.append(diTau_offline90Rate.GetBinContent(ibin))
    diTau_offline50Rate_list.append(diTau_offline50Rate.GetBinContent(ibin))


plt.figure(figsize=(10,10))
# plt.plot(Xs_list, singleTau_offline95Rate_list, linewidth=2, color='blue', label='Offline threshold @ 95%')
plt.plot(Xs_list, singleTau_offline90Rate_list, linewidth=2, color='red', label='Offline threshold @ 90%')
plt.plot(Xs_list, singleTau_offline50Rate_list, linewidth=2, color='green', label='Offline threshold @ 50%')
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=16)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel('Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(0,200)
myThr = np.interp(31, singleTau_offline90Rate_list[::-1], Xs_list[::-1])
myRate = np.interp(150, Xs_list, singleTau_offline90Rate_list)
plt.hlines(myRate, 0, 150, lw=2, color='dimgray')
plt.vlines(150, 1, 200, lw=2, color='dimgray')
plt.hlines(31, 0, myThr, lw=2, color='black')
plt.vlines(myThr, 1, 200, lw=2, color='black')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/rate_singleTau_offline_wp'+WP+'.pdf')
plt.close()

print('New Offline Threshold for SingleTau Rate 31kHz :', myThr)
print('New SingleTau Rate for Offline Threshold 150GeV :', myRate)

plt.figure(figsize=(10,10))
# plt.plot(Xs_list, diTau_offline95Rate_list, linewidth=2, color='blue', label='Offline threshold @ 95%')
plt.plot(Xs_list, diTau_offline90Rate_list, linewidth=2, color='red', label='Offline threshold @ 90%')
plt.plot(Xs_list, diTau_offline50Rate_list, linewidth=2, color='green', label='Offline threshold @ 50%')
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=16)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel('Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(0,200)
myThr = np.interp(33, diTau_offline90Rate_list[::-1], Xs_list[::-1])
myRate = np.interp(90, Xs_list, diTau_offline90Rate_list)
plt.hlines(myRate, 0, 90, lw=2, color='dimgray')
plt.vlines(90, 1, 200, lw=2, color='dimgray')
plt.hlines(33, 0, myThr, lw=2, color='black')
plt.vlines(myThr, 1, 200, lw=2, color='black')
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/rate_diTau_offline_wp'+WP+'.pdf')
plt.close()

print('New Offline Threshold for DoubleTau Rate 33kHz :', myThr)
print('New DoubleTau Rate for Offline Threshold 90GeV :', myRate)

