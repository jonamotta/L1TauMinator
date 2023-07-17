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
    parser.add_option('--CBCEsplit',        dest='CBCEsplit',      default=1.5, type=float)
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/rate'+tag)

    WP = options.WP

    if options.loop:
        #passing histograms (numerators)
        singleTau_offline90PtProgression = ROOT.TH1F("singleTau_offline90PtProgression","singleTau_offline90PtProgression",500,0.,500.)
        diTau_offline90PtProgression = ROOT.TH2F("diTau_offline90PtProgression","diTau_offline90PtProgression",500,0.,500.,500,0.,500.)
        singleTau_offline90Rate = ROOT.TH1F("singleTau_offline90Rate","singleTau_offline90Rate",500,0.,500.)
        diTau_offline90Rate = ROOT.TH1F("diTau_offline90Rate","diTau_offline90Rate",500,0.,500.)

        singleTau_offline90PtProgression_CB = ROOT.TH1F("singleTau_offline90PtProgression_CB","singleTau_offline90PtProgression_CB",500,0.,500.)
        diTau_offline90PtProgression_CB = ROOT.TH2F("diTau_offline90PtProgression_CB","diTau_offline90PtProgression_CB",500,0.,500.,500,0.,500.)
        singleTau_offline90Rate_CB = ROOT.TH1F("singleTau_offline90Rate_CB","singleTau_offline90Rate_CB",500,0.,500.)
        diTau_offline90Rate_CB = ROOT.TH1F("diTau_offline90Rate_CB","diTau_offline90Rate_CB",500,0.,500.)
        
        singleTau_offline90PtProgression_CE = ROOT.TH1F("singleTau_offline90PtProgression_CE","singleTau_offline90PtProgression_CE",500,0.,500.)
        diTau_offline90PtProgression_CE = ROOT.TH2F("diTau_offline90PtProgression_CE","diTau_offline90PtProgression_CE",500,0.,500.,500,0.,500.)
        singleTau_offline90Rate_CE = ROOT.TH1F("singleTau_offline90Rate_CE","singleTau_offline90Rate_CE",500,0.,500.)
        diTau_offline90Rate_CE = ROOT.TH1F("diTau_offline90Rate_CE","diTau_offline90Rate_CE",500,0.,500.)

        #denominator
        denominator = 0.

        # working points
        idWp_CB = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/TauMinator_CB_ident_plots/CLTW_TauIdentifier_WPs.pkl')
        idWp_CE = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/TauMinator_CE_ident_plots/CLTW_TauIdentifier_WPs.pkl')
        mapping_dict = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping'+options.WP+'.pkl')
        mapping_dict_CB = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping'+options.WP+'_CB.pkl')
        mapping_dict_CE = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping'+options.WP+'_CE.pkl')
        online_thresholds = range(0, 175, 1)

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/MinBias_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        # inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
        inChain = ROOT.TChain("L1CaloTauNtuplizerProducer/L1TauMinatorTree");
        inChain.Add(directory+'/Ntuple_*.root');
        nEntries = inChain.GetEntries()
        for evt in range(0, nEntries):
            if evt%1000==0: print('--> ',evt)
            # if evt == 100000: break

            entry = inChain.GetEntry(evt)

            denominator += 1.

            _l1tau_pt = list(inChain.minatedl1tau_pt)
            _l1tau_eta = list(inChain.minatedl1tau_eta)
            _l1tau_phi = list(inChain.minatedl1tau_phi)
            _l1tau_IDscore = list(inChain.minatedl1tau_IDscore)

            # filledProgression  = False
            PtObjsProgression = [ROOT.TLorentzVector(-1,0,0,0),ROOT.TLorentzVector(-1,0,0,0)]
            PtObjsProgression_CB = [ROOT.TLorentzVector(-1,0,0,0),ROOT.TLorentzVector(-1,0,0,0)]
            PtObjsProgression_CE = [ROOT.TLorentzVector(-1,0,0,0),ROOT.TLorentzVector(-1,0,0,0)]

            bestTauL1 = ROOT.TLorentzVector(-1,0,0,0)

            # loop over TauMinator taus
            for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_IDscore):
                l1tau = ROOT.TLorentzVector()
                l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                if abs(l1tau.Eta()) > options.etaEr: continue # skip taus out of acceptance

                if abs(l1tau.Eta()) < options.CBCEsplit:
                    if l1tauId < idWp_CB['wp'+WP]: continue
                else:
                    if l1tauId < idWp_CE['wp'+WP]: continue

                # single
                if l1tau.Pt() > bestTauL1.Pt():
                    bestTauL1 = l1tau

                # di
                if l1tau.Pt() >= PtObjsProgression[0].Pt():
                    PtObjsProgression[1] = PtObjsProgression[0]
                    PtObjsProgression[0] = l1tau
                
                elif l1tau.Pt() >= PtObjsProgression[1].Pt():
                    PtObjsProgression[1] = l1tau

                if abs(l1tau.Eta()) < 1.5:
                    if l1tau.Pt() >= PtObjsProgression_CB[0].Pt():
                        PtObjsProgression_CB[1] = PtObjsProgression_CB[0]
                        PtObjsProgression_CB[0] = l1tau
                    
                    elif l1tau.Pt() >= PtObjsProgression_CB[1].Pt():
                        PtObjsProgression_CB[1] = l1tau
                else:
                    if l1tau.Pt() >= PtObjsProgression_CE[0].Pt():
                        PtObjsProgression_CE[1] = PtObjsProgression_CE[0]
                        PtObjsProgression_CE[0] = l1tau
                    
                    elif l1tau.Pt() >= PtObjsProgression_CE[1].Pt():
                        PtObjsProgression_CE[1] = l1tau

            # single
            if abs(bestTauL1.Eta()) < 1.5:
                offlinePt = 1.498 * bestTauL1.Pt() - 1.676  #  mapping_dict_CB['wp'+WP+'_slope_pt90'] * bestTauL1.Pt() + mapping_dict_CB['wp'+WP+'_intercept_pt90']
                singleTau_offline90PtProgression.Fill(offlinePt)
                singleTau_offline90PtProgression_CB.Fill(offlinePt)
            else:
                offlinePt = 1.662 * bestTauL1.Pt() - 6.207  #  mapping_dict_CE['wp'+WP+'_slope_pt90'] * bestTauL1.Pt() + mapping_dict_CE['wp'+WP+'_intercept_pt90']
                singleTau_offline90PtProgression.Fill(offlinePt)
                singleTau_offline90PtProgression_CE.Fill(offlinePt)

            # di
            if PtObjsProgression[0].Pt()>=0 and PtObjsProgression[1].Pt()>=0:
                if abs(PtObjsProgression[1].Eta()) < 1.5:
                    offlinePt1 = 1.498 * PtObjsProgression[1].Pt() - 1.676  # mapping_dict['wp'+WP+'_slope_pt90'] * PtObjsProgression[1].Pt() + mapping_dict['wp'+WP+'_intercept_pt90']
                
                else:
                    offlinePt1 = 1.662 * PtObjsProgression[1].Pt() - 6.207  #mapping_dict_CE['wp'+WP+'_slope_pt90'] * PtObjsProgression[1].Pt() + mapping_dict_CE['wp'+WP+'_intercept_pt90']

                if abs(PtObjsProgression[0].Eta()) < 1.5:
                    offlinePt0 = 1.498 * PtObjsProgression[0].Pt() - 1.676  # mapping_dict['wp'+WP+'_slope_pt90'] * PtObjsProgression[0].Pt() + mapping_dict['wp'+WP+'_intercept_pt90']
                    diTau_offline90PtProgression.Fill(offlinePt0, offlinePt1)
                    diTau_offline90PtProgression_CB.Fill(offlinePt0, offlinePt1)
                
                else:
                    offlinePt0 = 1.662 * PtObjsProgression[0].Pt() - 6.207  #mapping_dict_CE['wp'+WP+'_slope_pt90'] * PtObjsProgression[0].Pt() + mapping_dict_CE['wp'+WP+'_intercept_pt90']
                    diTau_offline90PtProgression.Fill(offlinePt0, offlinePt1)
                    diTau_offline90PtProgression_CE.Fill(offlinePt0, offlinePt1)


        # end of the loop over the events
        #################################

        scale=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf
        scale=31038 # from menu

        for i in range(0,501):
            singleTau_offline90Rate.SetBinContent(i+1,singleTau_offline90PtProgression.Integral(i+1,501)/denominator*scale)
            singleTau_offline90Rate_CB.SetBinContent(i+1,singleTau_offline90PtProgression_CB.Integral(i+1,501)/denominator*scale)
            singleTau_offline90Rate_CE.SetBinContent(i+1,singleTau_offline90PtProgression_CE.Integral(i+1,501)/denominator*scale)
            
            diTau_offline90Rate.SetBinContent(i+1,diTau_offline90PtProgression.Integral(i+1,501,i+1,501)/denominator*scale)
            diTau_offline90Rate_CB.SetBinContent(i+1,diTau_offline90PtProgression_CB.Integral(i+1,501,i+1,501)/denominator*scale)
            diTau_offline90Rate_CE.SetBinContent(i+1,diTau_offline90PtProgression_CE.Integral(i+1,501,i+1,501)/denominator*scale)

        # save to file 
        fileout = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_wp"+WP+".root","RECREATE")
        singleTau_offline90PtProgression.Write()
        singleTau_offline90PtProgression_CB.Write()
        singleTau_offline90PtProgression_CE.Write()

        diTau_offline90PtProgression.Write()
        diTau_offline90PtProgression_CB.Write()
        diTau_offline90PtProgression_CE.Write()

        singleTau_offline90Rate.Write()
        singleTau_offline90Rate_CB.Write()
        singleTau_offline90Rate_CE.Write()

        diTau_offline90Rate.Write()
        diTau_offline90Rate_CB.Write()
        diTau_offline90Rate_CE.Write()

        fileout.Close()

    else:
        filein = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_wp"+WP+".root","READ")

        singleTau_offline90Rate = filein.Get("singleTau_offline90Rate")
        singleTau_offline90Rate_CB = filein.Get("singleTau_offline90Rate_CB")
        singleTau_offline90Rate_CE = filein.Get("singleTau_offline90Rate_CE")

        diTau_offline90Rate = filein.Get("diTau_offline90Rate")
        diTau_offline90Rate_CB = filein.Get("diTau_offline90Rate_CB")
        diTau_offline90Rate_CE = filein.Get("diTau_offline90Rate_CE")



Xs_list = []
singleTau_offline90Rate_list = []
singleTau_offline90Rate_CB_list = []
singleTau_offline90Rate_CE_list = []
diTau_offline90Rate_list = []
diTau_offline90Rate_CB_list = []
diTau_offline90Rate_CE_list = []
for ibin in range(1,singleTau_offline90Rate.GetNbinsX()+1):
    Xs_list.append(ibin)
    singleTau_offline90Rate_list.append(singleTau_offline90Rate.GetBinContent(ibin))
    singleTau_offline90Rate_CB_list.append(singleTau_offline90Rate_CB.GetBinContent(ibin))
    singleTau_offline90Rate_CE_list.append(singleTau_offline90Rate_CE.GetBinContent(ibin))

    diTau_offline90Rate_list.append(diTau_offline90Rate.GetBinContent(ibin))
    diTau_offline90Rate_CB_list.append(diTau_offline90Rate_CB.GetBinContent(ibin))
    diTau_offline90Rate_CE_list.append(diTau_offline90Rate_CE.GetBinContent(ibin))



plt.figure(figsize=(10,10))
plt.plot(Xs_list, singleTau_offline90Rate_list, linewidth=2, color='black', label=r'Full detector')
plt.plot(Xs_list, singleTau_offline90Rate_CB_list, linewidth=2, color='green', label='Barrel')
plt.plot(Xs_list, singleTau_offline90Rate_CE_list, linewidth=2, color='red', label='Endcap')
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=16)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
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

print('')
print('SingleTau Offline Rate @ 25GeV :', np.interp(25, Xs_list, singleTau_offline90Rate_list))
print('SingleTau Offline Rate @ 50GeV :', np.interp(50, Xs_list, singleTau_offline90Rate_list))
print('SingleTau Offline Rate @ 75GeV :', np.interp(75, Xs_list, singleTau_offline90Rate_list))
print('SingleTau Offline Rate @ 100GeV :', np.interp(100, Xs_list, singleTau_offline90Rate_list))
print('SingleTau Offline Rate @ 125GeV :', np.interp(125, Xs_list, singleTau_offline90Rate_list))
print('SingleTau Offline Rate @ 150GeV :', np.interp(150, Xs_list, singleTau_offline90Rate_list))
print('')
print('SingleTau Offline CB Rate @ 150GeV :', np.interp(150, Xs_list, singleTau_offline90Rate_CB_list))
print('SingleTau Offline CE Rate @ 150GeV :', np.interp(150, Xs_list, singleTau_offline90Rate_CE_list))
print('')

print('New Offline Threshold for SingleTau Rate 31kHz :', myThr)
print('New SingleTau Rate for Offline Threshold 150GeV :', myRate)

plt.figure(figsize=(10,10))
plt.plot(Xs_list, diTau_offline90Rate_list, linewidth=2, color='black', label=r'Full detector')
plt.plot(Xs_list, diTau_offline90Rate_CB_list, linewidth=2, color='green', label='Barrel')
plt.plot(Xs_list, diTau_offline90Rate_CE_list, linewidth=2, color='red', label='Endcap')
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=16)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Double-$\tau_{h}$ Rate [kHz]')
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

