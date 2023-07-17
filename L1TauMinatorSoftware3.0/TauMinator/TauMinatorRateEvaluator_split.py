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
    parser.add_option("--WP_CB",            dest="WP_CB",          default="90")
    parser.add_option("--WP_CE",            dest="WP_CE",          default="90")
    parser.add_option('--CBCEsplit',        dest='CBCEsplit',      default=1.55, type=float)
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/rate'+tag)

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
        singleTau_onlinePtProgression = ROOT.TH1F("singleTau_onlinePtProgression","singleTau_onlinePtProgression",500,0.,500.)
        singleTau_onlineRate = ROOT.TH1F("singleTau_onlineRate","singleTau_onlineRate",500,0.,500.)

        singleTau_onlinePtProgression_CB = ROOT.TH1F("singleTau_onlinePtProgression_CB","singleTau_onlinePtProgression_CB",500,0.,500.)
        singleTau_onlineRate_CB = ROOT.TH1F("singleTau_onlineRate_CB","singleTau_onlineRate_CB",500,0.,500.)
        
        singleTau_onlinePtProgression_CE = ROOT.TH1F("singleTau_onlinePtProgression_CE","singleTau_onlinePtProgression_CE",500,0.,500.)
        singleTau_onlineRate_CE = ROOT.TH1F("singleTau_onlineRate_CE","singleTau_onlineRate_CE",500,0.,500.)

        diTau_onlinePtProgression = ROOT.TH2F("diTau_onlinePtProgression","diTau_onlinePtProgression",500,0.,500.,500,0.,500.)
        diTau_onlineRate = ROOT.TH1F("diTau_onlineRate","diTau_onlineRate",500,0.,500.)

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
        mapping_dict = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping_CB'+WP_CB+'_CE'+WP_CE+'.pkl')
        mapping_dict_CB = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping_CB'+WP_CB+'.pkl')
        mapping_dict_CE = load_obj(perfdir+'/turnons'+tag+'/online2offline_mapping_CE'+WP_CE+'.pkl')
        online_thresholds = range(0, 175, 1)

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/MinBias_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        # inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
        inChain = ROOT.TChain("L1CaloTauNtuplizerProducer/L1TauMinatorTree");
        inChain.Add(directory+'/Ntuple_*.root');
        nEntries = inChain.GetEntries()
        print('Total number of entries :', nEntries)
        for evt in range(0, nEntries):
            if evt%1000==0: print('--> ',evt)
            # if evt == 5000: break

            entry = inChain.GetEntry(evt)

            denominator += 1.

            _l1tau_pt = list(inChain.minatedl1tau_pt)
            _l1tau_eta = list(inChain.minatedl1tau_eta)
            _l1tau_phi = list(inChain.minatedl1tau_phi)
            _l1tau_quality = list(inChain.minatedl1tau_quality)
            # _l1tau_quality = list(inChain.minatedl1tau_IDscore)

            # filledProgression  = False
            PtObjsProgression = [ROOT.TLorentzVector(-1,0,0,0),ROOT.TLorentzVector(-1,0,0,0)]
            PtObjsProgression_CB = [ROOT.TLorentzVector(-1,0,0,0),ROOT.TLorentzVector(-1,0,0,0)]
            PtObjsProgression_CE = [ROOT.TLorentzVector(-1,0,0,0),ROOT.TLorentzVector(-1,0,0,0)]

            bestTauL1 = ROOT.TLorentzVector(-1,0,0,0)

            # loop over TauMinator taus
            for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_quality):
                l1tau = ROOT.TLorentzVector()
                l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                if abs(l1tau.Eta()) > options.etaEr: continue # skip taus out of acceptance

                if abs(l1tau.Eta()) < options.CBCEsplit:
                    if l1tauId < qualityCut_CB: continue
                else:
                    if l1tauId < qualityCut_CE: continue

                # single
                if l1tau.Pt() > bestTauL1.Pt():
                    bestTauL1 = l1tau

                # di
                if l1tau.Pt() >= PtObjsProgression[0].Pt(): # and l1tau.DeltaR(PtObjsProgression[0])>0.5:
                    PtObjsProgression[1] = PtObjsProgression[0]
                    PtObjsProgression[0] = l1tau
                
                elif l1tau.Pt() >= PtObjsProgression[1].Pt(): # and l1tau.DeltaR(PtObjsProgression[1])>0.5:
                    PtObjsProgression[1] = l1tau

                if abs(l1tau.Eta()) < 1.5:
                    if l1tau.Pt() >= PtObjsProgression_CB[0].Pt(): # and l1tau.DeltaR(PtObjsProgression[0])>0.5:
                        PtObjsProgression_CB[1] = PtObjsProgression_CB[0]
                        PtObjsProgression_CB[0] = l1tau
                    
                    elif l1tau.Pt() >= PtObjsProgression_CB[1].Pt(): # and l1tau.DeltaR(PtObjsProgression[1])>0.5:
                        PtObjsProgression_CB[1] = l1tau
                else:
                    if l1tau.Pt() >= PtObjsProgression_CE[0].Pt(): # and l1tau.DeltaR(PtObjsProgression[0])>0.5:
                        PtObjsProgression_CE[1] = PtObjsProgression_CE[0]
                        PtObjsProgression_CE[0] = l1tau
                    
                    elif l1tau.Pt() >= PtObjsProgression_CE[1].Pt(): # and l1tau.DeltaR(PtObjsProgression[1])>0.5:
                        PtObjsProgression_CE[1] = l1tau

            # single
            if abs(bestTauL1.Eta()) < 1.5:
                offlinePt = mapping_dict_CB['CB'+WP_CB+'_slope_pt90'] * bestTauL1.Pt() + mapping_dict_CB['CB'+WP_CB+'_intercept_pt90']
                singleTau_offline90PtProgression.Fill(offlinePt)
                singleTau_offline90PtProgression_CB.Fill(offlinePt)

                singleTau_onlinePtProgression.Fill(bestTauL1.Pt())
                singleTau_onlinePtProgression_CB.Fill(bestTauL1.Pt())
            
            else:
                offlinePt = mapping_dict_CE['CE'+WP_CE+'_slope_pt90'] * bestTauL1.Pt() + mapping_dict_CE['CE'+WP_CE+'_intercept_pt90']
                singleTau_offline90PtProgression.Fill(offlinePt)
                singleTau_offline90PtProgression_CE.Fill(offlinePt)

                singleTau_onlinePtProgression.Fill(bestTauL1.Pt())
                singleTau_onlinePtProgression_CE.Fill(bestTauL1.Pt())

            # di
            if PtObjsProgression[0].Pt()>=0 and PtObjsProgression[1].Pt()>=0:
                diTau_onlinePtProgression.Fill(PtObjsProgression[0].Pt(), PtObjsProgression[1].Pt())

                if abs(PtObjsProgression[1].Eta()) < 1.5:
                    offlinePt1 = mapping_dict_CB['CB'+WP_CB+'_slope_pt90'] * PtObjsProgression[1].Pt() + mapping_dict_CB['CB'+WP_CB+'_intercept_pt90']
                
                else:
                    offlinePt1 = mapping_dict_CE['CE'+WP_CE+'_slope_pt90'] * PtObjsProgression[1].Pt() + mapping_dict_CE['CE'+WP_CE+'_intercept_pt90']

                if abs(PtObjsProgression[0].Eta()) < 1.5:
                    offlinePt0 = mapping_dict_CB['CB'+WP_CB+'_slope_pt90'] * PtObjsProgression[0].Pt() + mapping_dict_CB['CB'+WP_CB+'_intercept_pt90']
                    diTau_offline90PtProgression.Fill(offlinePt0, offlinePt1)
                    diTau_offline90PtProgression_CB.Fill(offlinePt0, offlinePt1)
                
                else:
                    offlinePt0 = mapping_dict_CE['CE'+WP_CE+'_slope_pt90'] * PtObjsProgression[0].Pt() + mapping_dict_CE['CE'+WP_CE+'_intercept_pt90']
                    diTau_offline90PtProgression.Fill(offlinePt0, offlinePt1)
                    diTau_offline90PtProgression_CE.Fill(offlinePt0, offlinePt1)


        # end of the loop over the events
        #################################

        scale=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf
        scale=31038 # from menu

        for i in range(0,501):
            singleTau_onlineRate.SetBinContent(i+1,singleTau_onlinePtProgression.Integral(i+1,501)/denominator*scale)
            singleTau_onlineRate_CB.SetBinContent(i+1,singleTau_onlinePtProgression_CB.Integral(i+1,501)/denominator*scale)
            singleTau_onlineRate_CE.SetBinContent(i+1,singleTau_onlinePtProgression_CE.Integral(i+1,501)/denominator*scale)

            diTau_onlineRate.SetBinContent(i+1,diTau_onlinePtProgression.Integral(i+1,501,i+1,501)/denominator*scale)

            singleTau_offline90Rate.SetBinContent(i+1,singleTau_offline90PtProgression.Integral(i+1,501)/denominator*scale)
            singleTau_offline90Rate_CB.SetBinContent(i+1,singleTau_offline90PtProgression_CB.Integral(i+1,501)/denominator*scale)
            singleTau_offline90Rate_CE.SetBinContent(i+1,singleTau_offline90PtProgression_CE.Integral(i+1,501)/denominator*scale)            

            diTau_offline90Rate.SetBinContent(i+1,diTau_offline90PtProgression.Integral(i+1,501,i+1,501)/denominator*scale)
            diTau_offline90Rate_CB.SetBinContent(i+1,diTau_offline90PtProgression_CB.Integral(i+1,501,i+1,501)/denominator*scale)
            diTau_offline90Rate_CE.SetBinContent(i+1,diTau_offline90PtProgression_CE.Integral(i+1,501,i+1,501)/denominator*scale)

        # save to file 
        fileout = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_CB"+WP_CB+"_CE"+WP_CE+".root","RECREATE")
        singleTau_onlinePtProgression.Write()
        singleTau_onlinePtProgression_CB.Write()
        singleTau_onlinePtProgression_CE.Write()

        diTau_onlinePtProgression.Write()

        singleTau_offline90PtProgression.Write()
        singleTau_offline90PtProgression_CB.Write()
        singleTau_offline90PtProgression_CE.Write()

        diTau_offline90PtProgression.Write()
        diTau_offline90PtProgression_CB.Write()
        diTau_offline90PtProgression_CE.Write()

        singleTau_onlineRate.Write()
        singleTau_onlineRate_CB.Write()
        singleTau_onlineRate_CE.Write()

        diTau_onlineRate.Write()

        singleTau_offline90Rate.Write()
        singleTau_offline90Rate_CB.Write()
        singleTau_offline90Rate_CE.Write()

        diTau_offline90Rate.Write()
        diTau_offline90Rate_CB.Write()
        diTau_offline90Rate_CE.Write()

        fileout.Close()

    else:
        filein = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_CB"+WP_CB+"_CE"+WP_CE+".root","READ")

        singleTau_onlineRate = filein.Get("singleTau_onlineRate")
        singleTau_onlineRate_CB = filein.Get("singleTau_onlineRate_CB")
        singleTau_onlineRate_CE = filein.Get("singleTau_onlineRate_CE")

        diTau_onlineRate = filein.Get("diTau_onlineRate")

        singleTau_offline90Rate = filein.Get("singleTau_offline90Rate")
        singleTau_offline90Rate_CB = filein.Get("singleTau_offline90Rate_CB")
        singleTau_offline90Rate_CE = filein.Get("singleTau_offline90Rate_CE")

        diTau_offline90Rate = filein.Get("diTau_offline90Rate")
        diTau_offline90Rate_CB = filein.Get("diTau_offline90Rate_CB")
        diTau_offline90Rate_CE = filein.Get("diTau_offline90Rate_CE")



Xs_list = []
singleTau_onlineRate_list = []
singleTau_onlineRate_CB_list = []
singleTau_onlineRate_CE_list = []
singleTau_offline90Rate_list = []
singleTau_offline90Rate_CB_list = []
singleTau_offline90Rate_CE_list = []
diTau_onlineRate_list = []
diTau_offline90Rate_list = []
diTau_offline90Rate_CB_list = []
diTau_offline90Rate_CE_list = []
for ibin in range(1,singleTau_offline90Rate.GetNbinsX()+1):
    Xs_list.append(ibin)
    
    singleTau_onlineRate_list.append(singleTau_onlineRate.GetBinContent(ibin))
    singleTau_onlineRate_CB_list.append(singleTau_onlineRate_CB.GetBinContent(ibin))
    singleTau_onlineRate_CE_list.append(singleTau_onlineRate_CE.GetBinContent(ibin))

    diTau_onlineRate_list.append(diTau_onlineRate.GetBinContent(ibin))

    singleTau_offline90Rate_list.append(singleTau_offline90Rate.GetBinContent(ibin))
    singleTau_offline90Rate_CB_list.append(singleTau_offline90Rate_CB.GetBinContent(ibin))
    singleTau_offline90Rate_CE_list.append(singleTau_offline90Rate_CE.GetBinContent(ibin))

    diTau_offline90Rate_list.append(diTau_offline90Rate.GetBinContent(ibin))
    diTau_offline90Rate_CB_list.append(diTau_offline90Rate_CB.GetBinContent(ibin))
    diTau_offline90Rate_CE_list.append(diTau_offline90Rate_CE.GetBinContent(ibin))



plt.figure(figsize=(10,10))
plt.plot(Xs_list, singleTau_offline90Rate_list, linewidth=2, color='#dd5129', label=r'Full detector')
plt.plot(Xs_list, singleTau_offline90Rate_CB_list, linewidth=2, color='#0f7ba2', label='Barrel')
plt.plot(Xs_list, singleTau_offline90Rate_CE_list, linewidth=2, color='#43b284', label='Endcap')
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
plt.savefig(perfdir+'/rate'+tag+'/rate_singleTau_offline_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
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
plt.plot(Xs_list, diTau_offline90Rate_list, linewidth=2, color='#dd5129', label=r'Full detector')
plt.plot(Xs_list, diTau_offline90Rate_CB_list, linewidth=2, color='#0f7ba2', label='Barrel')
plt.plot(Xs_list, diTau_offline90Rate_CE_list, linewidth=2, color='#43b284', label='Endcap')
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
plt.savefig(perfdir+'/rate'+tag+'/rate_diTau_offline_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
plt.close()

print('New Offline Threshold for DoubleTau Rate 33kHz :', myThr)
print('New DoubleTau Rate for Offline Threshold 90GeV :', myRate)







WP_CB_text = ""
WP_CE_text = ""
if WP_CB == "90": WP_CB_text = 'Tight'
if WP_CE == "90": WP_CE_text = 'Tight'
if WP_CB == "95": WP_CB_text = 'Medium'
if WP_CE == "95": WP_CE_text = 'Medium'
if WP_CB == "99": WP_CB_text = 'Loose'
if WP_CE == "99": WP_CE_text = 'Loose'


plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], singleTau_offline90Rate_list[::3], lw=3, marker='o', markersize='10', color='#d04e00', label=r'TauMinator ('+WP_CB_text+'-'+WP_CE_text+' WP)', zorder=0)
plt.plot(Xs_list[::3], singleTau_offline90Rate_CB_list[::3], lw=3, marker='s', markersize='10', color='#f6c200', label=r'TauMinator ('+WP_CB_text+' WP) - Barrel', zorder=1)
plt.plot(Xs_list[::3], singleTau_offline90Rate_CE_list[::3], lw=3, marker='^', markersize='10', color='#0086a8', label=r'TauMinator ('+WP_CE_text+' WP) - Endcap', zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(5,3E4)
plt.xlim(25,175)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], singleTau_onlineRate_list[::3], lw=3, marker='o', markersize='10', color='#d04e00', label=r'TauMinator ('+WP_CB_text+'-'+WP_CE_text+' WP)', zorder=0)
plt.plot(Xs_list[::3], singleTau_onlineRate_CB_list[::3], lw=3, marker='s', markersize='10', color='#f6c200', label=r'TauMinator ('+WP_CB_text+' WP) - Barrel', zorder=1)
plt.plot(Xs_list[::3], singleTau_onlineRate_CE_list[::3], lw=3, marker='^', markersize='10', color='#0086a8', label=r'TauMinator ('+WP_CE_text+' WP) - Endcap', zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,3E4)
plt.xlim(10,175)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_online_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], diTau_offline90Rate_list[::3], lw=3, marker='o', markersize='10', color='#d04e00', label=r'TauMinator ('+WP_CB_text+'-'+WP_CE_text+' WP)', zorder=0)
plt.plot(Xs_list[::3], diTau_offline90Rate_CB_list[::3], lw=3, marker='s', markersize='10', color='#f6c200', label=r'TauMinator ('+WP_CB_text+' WP) - Barrel', zorder=1)
plt.plot(Xs_list[::3], diTau_offline90Rate_CE_list[::3], lw=3, marker='^', markersize='10', color='#0086a8', label=r'TauMinator ('+WP_CE_text+' WP) - Endcap', zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Double-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,1E4)
plt.xlim(25,135)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_diTau_offline_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], singleTau_onlineRate_list[::3], lw=3, marker='o', markersize='10', color='#d04e00', label=r'Single-$\tau_{h}$ TauMinator ('+WP_CB_text+'-'+WP_CE_text+' WP)', zorder=0)
plt.plot(Xs_list[::3], diTau_onlineRate_list[::3], lw=3, marker='o', markersize='10', color='#f6c200', label=r'Double-$\tau_{h}$ TauMinator ('+WP_CB_text+'-'+WP_CE_text+' WP)', zorder=0)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Rate [kHz]')
plt.ylim(1,3E4)
plt.xlim(10,175)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_diTau_online_CB'+WP_CB+'_CE'+WP_CE+'.pdf')
plt.close()



fileinTight = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_CB90_CE90.root","READ")
fileinMedium = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_CB95_CE95.root","READ")
fileinLoose = ROOT.TFile(perfdir+"/rate"+tag+"/rate_graphs"+tag+"_er"+str(options.etaEr)+"_CB99_CE99.root","READ")
rateTight = fileinTight.Get("singleTau_offline90Rate")
rateMedium = fileinMedium.Get("singleTau_offline90Rate")
rateLoose = fileinLoose.Get("singleTau_offline90Rate")
di_rateTight = fileinTight.Get("diTau_offline90Rate")
di_rateMedium = fileinMedium.Get("diTau_offline90Rate")
di_rateLoose = fileinLoose.Get("diTau_offline90Rate")

online_rateTight = fileinTight.Get("singleTau_onlineRate")
online_rateMedium = fileinMedium.Get("singleTau_onlineRate")
online_rateLoose = fileinLoose.Get("singleTau_onlineRate")

Xs_list = []
singleTau_onlineRate_Tight = []
singleTau_onlineRate_Medium = []
singleTau_onlineRate_Loose = []
singleTau_offline90Rate_Tight = []
singleTau_offline90Rate_Medium = []
singleTau_offline90Rate_Loose = []
diTau_offline90Rate_Tight = []
diTau_offline90Rate_Medium = []
diTau_offline90Rate_Loose = []
for ibin in range(1,singleTau_offline90Rate.GetNbinsX()+1):
    Xs_list.append(ibin)
    
    singleTau_onlineRate_Tight.append(online_rateTight.GetBinContent(ibin))
    singleTau_onlineRate_Medium.append(online_rateMedium.GetBinContent(ibin))
    singleTau_onlineRate_Loose.append(online_rateLoose.GetBinContent(ibin))

    singleTau_offline90Rate_Tight.append(rateTight.GetBinContent(ibin))
    singleTau_offline90Rate_Medium.append(rateMedium.GetBinContent(ibin))
    singleTau_offline90Rate_Loose.append(rateLoose.GetBinContent(ibin))

    diTau_offline90Rate_Tight.append(di_rateTight.GetBinContent(ibin))
    diTau_offline90Rate_Medium.append(di_rateMedium.GetBinContent(ibin))
    diTau_offline90Rate_Loose.append(di_rateLoose.GetBinContent(ibin))

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], singleTau_offline90Rate_Tight[::3], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(Xs_list[::3], singleTau_offline90Rate_Medium[::3], lw=3, marker='s', markersize='10', label=r'TauMinator (Medium-Medium WP)', color="#f6c200", zorder=1)
plt.plot(Xs_list[::3], singleTau_offline90Rate_Loose[::3], lw=3, marker='^', markersize='10', label=r'TauMinator (Loose-Loose WP)', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(10,4E4)
plt.xlim(25,175)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], singleTau_onlineRate_Tight[::3], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(Xs_list[::3], singleTau_onlineRate_Medium[::3], lw=3, marker='s', markersize='10', label=r'TauMinator (Medium-Medium WP)', color="#f6c200", zorder=1)
plt.plot(Xs_list[::3], singleTau_onlineRate_Loose[::3], lw=3, marker='^', markersize='10', label=r'TauMinator (Loose-Loose WP)', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,1E5)
plt.xlim(10,175)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_online.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3], diTau_offline90Rate_Tight[::3], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(Xs_list[::3], diTau_offline90Rate_Medium[::3], lw=3, marker='s', markersize='10', label=r'TauMinator (Medium-Medium WP)', color="#f6c200", zorder=1)
plt.plot(Xs_list[::3], diTau_offline90Rate_Loose[::3], lw=3, marker='^', markersize='10', label=r'TauMinator (Loose-Loose WP)', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Double-$\tau_{h}$ Rate [kHz]')
plt.ylim(5,1E4)
plt.xlim(25,135)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_diTau_offline.pdf')
plt.close()


# REDUCED STATISTICS, PLOT SCALING
off = {}
off['xs'] = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0]
off['CaloTau'] = [31038.0, 31038.0, 31038.0, 31037.2, 30964.4, 29954.5, 26175.4, 18478.7, 11238.3, 5801.2, 3260.8, 1851.0, 1175.4, 772.9, 547.2, 391.4, 297.8, 230.3, 183.8, 152.1, 119.9, 96.8, 79.8, 65.9, 56.4, 48.9, 41.2, 35.8, 31.4, 26.5, 22.9, 20.1, 18.5, 16.2, 14.4, 13.6, 11.8, 10.6, 9.8, 6.9, 5.9, 4.9]
off['CaloTauBarrel'] = [30976.2, 30968.0, 30441.4, 26076.3, 16480.5, 8442.8, 4481.2, 2509.0, 1527.5, 972.1, 670.9, 468.4, 347.2, 266.1, 213.6, 167.5, 135.4, 110.2, 91.9, 80.3, 66.4, 54.6, 46.3, 38.6, 33.5, 28.1, 23.9, 21.6, 19.0, 16.2, 13.9, 12.9, 11.8, 10.0, 8.5, 8.2, 7.5, 6.4, 6.4, 5.1, 4.1, 3.6]
off['CaloTauEndcap'] = [31038.0, 31038.0, 31038.0, 31036.7, 30935.8, 29764.8, 25661.7, 17657.4, 10422.2, 5123.3, 2744.0, 1469.0, 887.4, 551.0, 366.0, 250.2, 183.2, 136.1, 105.3, 83.1, 63.3, 51.0, 40.7, 32.9, 28.3, 23.9, 20.1, 17.0, 14.9, 12.4, 10.8, 8.8, 7.7, 6.9, 6.7, 6.2, 5.1, 4.9, 4.1, 2.6, 2.3, 1.8]
onl = {}
onl['xs'] = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0]
onl['CaloTau'] = [31038.0, 31031.8, 29149.7, 16486.7, 5947.4, 2329.9, 1129.8, 612.5, 378.1, 257.4, 174.2, 129.5, 100.6, 75.2, 57.1, 42.2, 33.7, 27.3, 22.6, 17.5, 14.4, 12.1, 9.3, 8.5, 6.9, 6.2, 4.1, 3.1, 2.6, 2.6, 2.6, 2.6, 1.8, 1.8, 1.5, 1.3, 1.3, 1.0, 1.0, 0.5, 0.5, 0.5]
onl['CaloTauBarrel'] = [30976.2, 30294.0, 20496.9, 7786.0, 3077.8, 1404.2, 756.6, 435.7, 283.4, 201.5, 142.1, 106.3, 82.9, 62.5, 47.9, 35.8, 28.6, 22.9, 19.0, 15.2, 12.9, 11.1, 8.2, 7.5, 6.4, 5.9, 4.1, 3.1, 2.6, 2.6, 2.6, 2.6, 1.8, 1.8, 1.5, 1.3, 1.3, 1.0, 1.0, 0.5, 0.5, 0.5]
onl['CaloTauEndcap'] = [31038.0, 30992.7, 27057.6, 12483.4, 3458.4, 1078.9, 439.8, 216.2, 122.5, 74.6, 46.3, 31.4, 23.7, 16.7, 12.4, 9.0, 7.2, 6.4, 5.1, 3.3, 2.3, 1.8, 1.5, 1.3, 0.8, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

# FULL STATISTICS, TABLE SCALING
# off = {}
# off['xs'] = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0]
# off['CaloTau'] =  [31038.0, 31036.5, 30787.0, 27221.8, 17852.7, 9475.5, 5101.2, 2854.7, 1752.4, 1136.8, 773.7, 550.1, 404.1, 303.0, 232.3, 185.0, 149.3, 121.4, 101.7, 85.7, 72.2, 60.2, 52.0, 44.5, 38.2, 33.1, 28.5, 25.0, 22.7, 20.4, 17.8, 16.0, 14.7, 13.1, 11.7, 10.3, 9.2, 8.1, 7.4, 6.7, 6.2, 5.7]
# off['CaloTauBarrel'] =  [30969.0, 30957.9, 30402.2, 25943.2, 16315.7, 8412.5, 4481.9, 2503.5, 1532.3, 994.6, 679.2, 483.6, 356.2, 265.9, 204.6, 162.8, 131.0, 106.3, 89.4, 75.4, 63.7, 53.2, 45.5, 38.8, 33.2, 28.8, 24.8, 22.0, 20.2, 18.3, 16.1, 14.5, 13.2, 11.8, 10.6, 9.4, 8.4, 7.3, 6.6, 6.0, 5.5, 4.9]
# off['CaloTauEndcap'] = [30916.8, 28585.4, 21042.5, 10285.9, 4178.8, 1803.5, 893.0, 486.8, 300.4, 195.3, 131.4, 93.6, 69.3, 54.1, 41.7, 33.2, 27.1, 21.8, 18.1, 15.2, 12.5, 10.2, 9.2, 8.0, 6.9, 6.0, 4.9, 4.0, 3.5, 3.0, 2.4, 2.2, 2.1, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0]

# onl = {}
# onl['xs'] = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0]
# onl['CaloTau'] =  [31038.0, 31032.1, 29165.8, 16518.5, 5945.9, 2347.0, 1145.2, 637.3, 388.7, 253.5, 177.4, 128.7, 97.9, 75.0, 58.4, 45.2, 36.3, 29.1, 24.9, 20.8, 17.1, 14.5, 12.0, 10.3, 8.4, 7.3, 6.3, 5.5, 5.0, 4.6, 4.2, 3.7, 3.3, 3.0, 2.7, 2.4, 2.2, 2.1, 1.9, 1.5, 1.3, 1.2]
# onl['CaloTauBarrel'] =  [30969.0, 30292.0, 20521.6, 7834.7, 3063.4, 1419.7, 762.7, 448.8, 285.9, 192.1, 136.6, 100.6, 77.1, 59.8, 46.5, 36.4, 29.0, 23.3, 20.2, 17.2, 14.4, 12.4, 10.4, 8.8, 7.1, 6.2, 5.4, 4.7, 4.3, 3.9, 3.5, 3.0, 2.7, 2.4, 2.1, 2.0, 1.9, 1.8, 1.6, 1.3, 1.1, 1.0]
# onl['CaloTauEndcap'] = [31038.0, 30989.9, 27051.2, 12537.4, 3469.8, 1085.5, 453.8, 228.8, 128.3, 78.9, 54.2, 37.5, 27.5, 19.9, 15.6, 11.5, 9.5, 7.7, 6.2, 4.7, 3.7, 3.0, 2.3, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.6, 0.5, 0.5, 0.4, 0.3, 0.3, 0.3]


plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_offline90Rate_Tight[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], off['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(25,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_comparison_tight.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_offline90Rate_Tight[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], off['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,500)
plt.xlim(90,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_comparison_tight_xReduced.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_offline90Rate_Medium[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Medium-Medium WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], off['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(25,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_comparison_medium.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_offline90Rate_Loose[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Loose-Loose WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], off['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(25,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_comparison_loose.pdf')
plt.close()









plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_onlineRate_Tight[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], onl['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(10,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_online_comparison_tight.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_onlineRate_CB_list[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], onl['CaloTauBarrel'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', title='Barrel', title_fontsize=20, fontsize=20)
legend._legend_box.align = "left"
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(10,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_online_comparison_barrel_CB'+WP_CB+'CE'+WP_CE+'.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_onlineRate_CE_list[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], onl['CaloTauEndcap'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', title='Endcap', title_fontsize=20, fontsize=20)
legend._legend_box.align = "left"
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(10,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_online_comparison_endcap_CB'+WP_CB+'CE'+WP_CE+'.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_offline90Rate_CB_list[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], off['CaloTauBarrel'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', title='Barrel', title_fontsize=20, fontsize=20)
legend._legend_box.align = "left"
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(25,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_comparison_barrel_CB'+WP_CB+'CE'+WP_CE+'.pdf')
plt.close()

plt.figure(figsize=(12,12))
plt.plot(Xs_list[::3][:68], singleTau_offline90Rate_CE_list[::3][:68], lw=3, marker='o', markersize='10', label=r'TauMinator (Tight-Tight WP)', color="#d04e00", zorder=0)
plt.plot(off['xs'], off['CaloTauEndcap'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="#0086a8", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', title='Endcap', title_fontsize=20, fontsize=20)
legend._legend_box.align = "left"
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,4E4)
plt.xlim(25,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_comparison_endcap_CB'+WP_CB+'CE'+WP_CE+'.pdf')
plt.close()












plt.figure(figsize=(12,12))
plt.plot(off['xs'], onl['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="black", zorder=2)
plt.plot(off['xs'], onl['CaloTauBarrel'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="red", zorder=2)
plt.plot(off['xs'], onl['CaloTauEndcap'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="blue", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', title_fontsize=20, fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Online threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,5E5)
plt.xlim(10,200)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_online_calotau.pdf')
plt.close()


plt.figure(figsize=(12,12))
plt.plot(off['xs'], off['CaloTau'], lw=3, marker='^', markersize='10', label=r'Calo Tau', color="black", zorder=2)
plt.plot(off['xs'], off['CaloTauBarrel'], lw=3, marker='^', markersize='10', label=r'Calo Tau Barrel', color="red", zorder=2)
plt.plot(off['xs'], off['CaloTauEndcap'], lw=3, marker='^', markersize='10', label=r'Calo Tau Endcap', color="blue", zorder=2)
plt.yscale("log")
plt.ylim(bottom=1)
legend = plt.legend(loc = 'upper right', title_fontsize=20, fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Offline threshold [GeV]')
plt.ylabel(r'Single-$\tau_{h}$ Rate [kHz]')
plt.ylim(1,5E5)
plt.xlim(10,160)
mplhep.cms.label('Phase-2 Simulation Preliminary', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/rate'+tag+'/DP_rate_singleTau_offline_calotau.pdf')
plt.close()





