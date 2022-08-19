from optparse import OptionParser
from itertools import chain
import pandas as pd
import numpy as np
import argparse
import uproot3
import glob
import sys
import os


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--v",          dest="v",          help="Version of the iteration",                                        default=None)
    parser.add_option("--date",       dest="date",       help="Date of birth of this version",                                   default=None)
    parser.add_option('--doHH',       dest='doHH',       help='Read the HH samples?',                       action='store_true', default=False)
    parser.add_option('--doQCD',      dest='doQCD',      help='Read the QCD samples?',                      action='store_true', default=False)
    parser.add_option('--doVBFH',     dest='doVBFH',     help='Read the VBF H samples?',                    action='store_true', default=False)
    parser.add_option('--doMinBias',  dest='doMinBias',  help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doTestRun',  dest='doTestRun',  help='Do test run with reduced number of events?', action='store_true', default=False)
    parser.add_option('--caloClNxM',  dest='caloClNxM',  help='Which shape of CaloCluster to use?',                              default="9x9")
    parser.add_option("--chunk_size", dest="chunk_size", help="Number of events per DF?",                   type=int,            default=5000)
    (options, args) = parser.parse_args()

    if not options.date or not options.v:
        print('** WARNING : no version and date specified --> no output folder specified')
        print('** EXITING')
        exit()

    if not options.doHH and not options.doQCD and not options.doVBFH and not options.doMinBias and not options.doTestRun:
        print('** WARNING : no matching dataset specified. What do you want to do (doHH, doQCD, doVBFH, doMinBias, doTestRun)?')
        print('** EXITING')
        exit()


    ##################### DEFINE INPUTS AND OUTPUTS ####################
    indir  = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples'
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

    if options.doHH:
        indir  += '/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW'
        outdir = outdir+'/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches'

    elif options.doVBFH:
        indir  += '/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW'
        outdir = outdir+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches'

    elif options.doQCD:
        indir  += '/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW'
        outdir = outdir+'/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches'

    elif options.doMinBias:
        indir  += '/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW'
        outdir = outdir+'/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches'

    elif options.doTestRun:
        indir  += '/test'
        outdir = outdir+'/test__batches'

    os.system('mkdir -p '+outdir+'/L1Clusters')
    os.system('mkdir -p '+outdir+'/GenObjects')

    print(options)

    # list Ntuples
    InFiles = []
    files = glob.glob(indir+'/Ntuple*.root')
    for file in files:
        InFiles.append(file)
    InFiles.sort()

    key = 'Ntuplizer/L1TauMinatorTree'
    branches_event  = ['EventNumber']
    branches_gentau = ['tau_Idx', 'tau_eta', 'tau_phi', 'tau_pt', 'tau_e', 'tau_m', 'tau_visEta', 'tau_visPhi', 'tau_visPt', 'tau_visE', 'tau_visM', 'tau_visPtEm', 'tau_visPtHad', 'tau_visEEm', 'tau_visEHad', 'tau_DM']
    branches_genjet = ['jet_Idx', 'jet_eta', 'jet_phi', 'jet_pt', 'jet_e', 'jet_eEm', 'jet_eHad', 'jet_eInv']
    branches_cl3d   = ['cl3d_pt', 'cl3d_energy', 'cl3d_eta', 'cl3d_phi', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_quality', 'cl3d_tauMatchIdx', 'cl3d_jetMatchIdx']
    NxM = options.caloClNxM
    branches_clNxM  = ['cl'+NxM+'_barrelSeeded', 'cl'+NxM+'_nHits', 'cl'+NxM+'_seedIeta', 'cl'+NxM+'_seedIphi', 'cl'+NxM+'_seedEta', 'cl'+NxM+'_seedPhi', 'cl'+NxM+'_isBarrel', 'cl'+NxM+'_isOverlap', 'cl'+NxM+'_isEndcap', 'cl'+NxM+'_tauMatchIdx', 'cl'+NxM+'_jetMatchIdx', 'cl'+NxM+'_totalEm', 'cl'+NxM+'_totalHad', 'cl'+NxM+'_totalEt', 'cl'+NxM+'_totalIem', 'cl'+NxM+'_totalIhad', 'cl'+NxM+'_totalIet', 'cl'+NxM+'_towerEta', 'cl'+NxM+'_towerPhi', 'cl'+NxM+'_towerEm', 'cl'+NxM+'_towerHad', 'cl'+NxM+'_towerEt', 'cl'+NxM+'_towerIeta', 'cl'+NxM+'_towerIphi', 'cl'+NxM+'_towerIem', 'cl'+NxM+'_towerIhad', 'cl'+NxM+'_towerIet', 'cl'+NxM+'_nEGs', 'cl'+NxM+'_towerEgEt', 'cl'+NxM+'_towerEgIet', 'cl'+NxM+'_towerNeg']

    # create file to store tags of batches
    tagsFile = open (outdir+'/tagsFile.txt', 'w')

    for i, Infile in enumerate(InFiles[:]):
        print(Infile)

        tag = Infile.split('/Ntuple')[1].split('.r')[0]

        # define the two paths where to store the hdf5 files
        saveTo = {
            'HGClus'  : outdir+'/L1Clusters/HGClus',
            'TowClus' : outdir+'/L1Clusters/TowClus'+NxM,
            'GenTaus' : outdir+'/GenObjects/GenTaus',
            'GenJets' : outdir+'/GenObjects/GenJets'
        }

        ##################### READ THE TTREES ####################
        TTree = uproot3.open(Infile)[key]

        arr_event  = TTree.arrays(branches_event)
        arr_gentau = TTree.arrays(branches_gentau)
        arr_genjet = TTree.arrays(branches_genjet)
        arr_cl3d   = TTree.arrays(branches_cl3d)
        arr_clNxM  = TTree.arrays(branches_clNxM)

        # print('*******************************************************')
        # bNxM = NxM.encode('utf-8')
        # print(len(arr_event[b'EventNumber']))
        # print(len(arr_gentau[b'tau_eta']))
        # for i in range(len(arr_gentau[b'tau_eta'])):
        #     print('    ->', len(arr_gentau[b'tau_eta'][i]))
        #     for j in range(len(arr_gentau[b'tau_eta'][i])):
        #         print('        - idx', arr_gentau[b'tau_Idx'][i][j],' eta', arr_gentau[b'tau_eta'][i][j], 'phi', arr_gentau[b'tau_phi'][i][j], 'pt', arr_gentau[b'tau_pt'][i][j])
        # print(len(arr_genjet[b'jet_eta']))
        # for i in range(len(arr_genjet[b'jet_eta'])):
        #     print('    ->', len(arr_genjet[b'jet_eta'][i]))
        # print(len(arr_cl3d[b'cl3d_pt']))
        # for i in range(len(arr_cl3d[b'cl3d_pt'])):
        #     print('    ->', len(arr_cl3d[b'cl3d_pt'][i]))
        # print(len(arr_clNxM[b'cl'+bNxM+b'_nHits']))
        # for i in range(len(arr_clNxM[b'cl'+bNxM+b'_nHits'])):
        #     print('    ->', len(arr_clNxM[b'cl'+bNxM+b'_nHits'][i]))
        # for i in range(len(arr_clNxM[b'cl'+bNxM+b'_towerEta'][0])):
        #     print('        ->', len(arr_clNxM[b'cl'+bNxM+b'_towerEta'][0][i]))
        # print('*******************************************************')

        df_event  = pd.DataFrame(arr_event)
        df_gentau = pd.DataFrame(arr_gentau)
        df_genjet = pd.DataFrame(arr_genjet)
        df_cl3d   = pd.DataFrame(arr_cl3d)
        df_clNxM  = pd.DataFrame(arr_clNxM)

        dfHGClus  = pd.concat([df_event, df_cl3d], axis=1)
        dfTowClus = pd.concat([df_event, df_clNxM], axis=1)
        dfGenTaus = pd.concat([df_event, df_gentau], axis=1)
        dfGenJets = pd.concat([df_event, df_genjet], axis=1)


        ##################### FLATTEN THE TTREES ####################

        # flatten out the jets dataframe
        dfFlatGenTaus = pd.DataFrame({
            'event'        : np.repeat(dfGenTaus[b'EventNumber'].values, dfGenTaus[b'tau_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'tau_Idx'      : list(chain.from_iterable(dfGenTaus[b'tau_Idx'])),
            'tau_eta'      : list(chain.from_iterable(dfGenTaus[b'tau_eta'])),
            'tau_phi'      : list(chain.from_iterable(dfGenTaus[b'tau_phi'])),
            'tau_pt'       : list(chain.from_iterable(dfGenTaus[b'tau_pt'])),
            'tau_e'        : list(chain.from_iterable(dfGenTaus[b'tau_e'])),
            'tau_m'        : list(chain.from_iterable(dfGenTaus[b'tau_m'])),
            'tau_visEta'   : list(chain.from_iterable(dfGenTaus[b'tau_visEta'])),
            'tau_visPhi'   : list(chain.from_iterable(dfGenTaus[b'tau_visPhi'])),
            'tau_visPt'    : list(chain.from_iterable(dfGenTaus[b'tau_visPt'])),
            'tau_visE'     : list(chain.from_iterable(dfGenTaus[b'tau_visE'])),
            'tau_visM'     : list(chain.from_iterable(dfGenTaus[b'tau_visM'])),
            'tau_visPtEm'  : list(chain.from_iterable(dfGenTaus[b'tau_visPtEm'])),
            'tau_visPtHad' : list(chain.from_iterable(dfGenTaus[b'tau_visPtHad'])),
            'tau_visEEm'   : list(chain.from_iterable(dfGenTaus[b'tau_visEEm'])),
            'tau_visEHad'  : list(chain.from_iterable(dfGenTaus[b'tau_visEHad'])),
            'tau_DM'       : list(chain.from_iterable(dfGenTaus[b'tau_DM']))
            })

        # flatten out the jets dataframe
        dfFlatGenJets = pd.DataFrame({
            'event'    : np.repeat(dfGenJets[b'EventNumber'].values, dfGenJets[b'jet_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'jet_Idx'  : list(chain.from_iterable(dfGenJets[b'jet_Idx'])),
            'jet_eta'  : list(chain.from_iterable(dfGenJets[b'jet_eta'])),
            'jet_phi'  : list(chain.from_iterable(dfGenJets[b'jet_phi'])),
            'jet_pt'   : list(chain.from_iterable(dfGenJets[b'jet_pt'])),
            'jet_e'    : list(chain.from_iterable(dfGenJets[b'jet_e'])),
            'jet_eEm'  : list(chain.from_iterable(dfGenJets[b'jet_eEm'])),
            'jet_eHad' : list(chain.from_iterable(dfGenJets[b'jet_eHad'])),
            'jet_eInv' : list(chain.from_iterable(dfGenJets[b'jet_eInv']))
            })

        # flatten out the hgcal clusters dataframe
        dfFlatHGClus = pd.DataFrame({
            'event'                 : np.repeat(dfHGClus[b'EventNumber'].values, dfHGClus[b'cl3d_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'cl3d_pt'               : list(chain.from_iterable(dfHGClus[b'cl3d_pt'])),
            'cl3d_energy'           : list(chain.from_iterable(dfHGClus[b'cl3d_energy'])),
            'cl3d_eta'              : list(chain.from_iterable(dfHGClus[b'cl3d_eta'])),
            'cl3d_phi'              : list(chain.from_iterable(dfHGClus[b'cl3d_phi'])),
            'cl3d_showerlength'     : list(chain.from_iterable(dfHGClus[b'cl3d_showerlength'])),
            'cl3d_coreshowerlength' : list(chain.from_iterable(dfHGClus[b'cl3d_coreshowerlength'])),
            'cl3d_firstlayer'       : list(chain.from_iterable(dfHGClus[b'cl3d_firstlayer'])),
            'cl3d_seetot'           : list(chain.from_iterable(dfHGClus[b'cl3d_seetot'])),
            'cl3d_seemax'           : list(chain.from_iterable(dfHGClus[b'cl3d_seemax'])),
            'cl3d_spptot'           : list(chain.from_iterable(dfHGClus[b'cl3d_spptot'])),
            'cl3d_sppmax'           : list(chain.from_iterable(dfHGClus[b'cl3d_sppmax'])),
            'cl3d_szz'              : list(chain.from_iterable(dfHGClus[b'cl3d_szz'])),
            'cl3d_srrtot'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrtot'])),
            'cl3d_srrmax'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrmax'])),
            'cl3d_srrmean'          : list(chain.from_iterable(dfHGClus[b'cl3d_srrmean'])),
            'cl3d_hoe'              : list(chain.from_iterable(dfHGClus[b'cl3d_hoe'])),
            'cl3d_meanz'            : list(chain.from_iterable(dfHGClus[b'cl3d_meanz'])),
            'cl3d_quality'          : list(chain.from_iterable(dfHGClus[b'cl3d_quality'])),
            'cl3d_tauMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_tauMatchIdx'])),
            'cl3d_jetMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_jetMatchIdx']))
            })

        bNxM = NxM.encode('utf-8')
        # flatten out the tower clusters dataframe
        dfFlatTowClus = pd.DataFrame({
            'event'           : np.repeat(dfTowClus[b'EventNumber'].values, dfTowClus[b'cl'+bNxM+b'_seedIeta'].str.len()), # event IDs are copied to keep proper track of what is what
            'cl_barrelSeeded' : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_barrelSeeded'])),
            'cl_nHits'        : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_nHits'])),
            'cl_nEGs'         : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_nEGs'])),
            'cl_seedIeta'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedIeta'])),
            'cl_seedIphi'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedIphi'])),
            'cl_seedEta'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedEta'])),
            'cl_seedPhi'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedPhi'])),
            'cl_isBarrel'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isBarrel'])),
            'cl_isOverlap'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isOverlap'])),
            'cl_isEndcap'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isEndcap'])),
            'cl_tauMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_tauMatchIdx'])),
            'cl_jetMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_jetMatchIdx'])),
            'cl_totalEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEm'])),
            'cl_totalHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalHad'])),
            'cl_totalEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEt'])),
            'cl_totalIem'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIem'])),
            'cl_totalIhad'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIhad'])),
            'cl_totalIet'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIet'])),
            'cl_towerEta'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEta'])),
            'cl_towerPhi'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerPhi'])),
            'cl_towerEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEm'])),
            'cl_towerHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerHad'])),
            'cl_towerEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEt'])),
            'cl_towerEgEt'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgEt'])),
            'cl_towerIeta'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIeta'])),
            'cl_towerIphi'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIphi'])),
            'cl_towerIem'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIem'])),
            'cl_towerIhad'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIhad'])),
            'cl_towerIet'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIet'])),
            'cl_towerEgIet'   : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgIet'])),
            'cl_towerNeg'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerNeg'])),
            })


        ##################### SAVE TO FILE ####################
        
        dfFlatHGClus.to_pickle(saveTo['HGClus']+tag+'.pkl')
        dfFlatTowClus.to_pickle(saveTo['TowClus']+tag+'.pkl')
        dfFlatGenTaus.to_pickle(saveTo['GenTaus']+tag+'.pkl')
        dfFlatGenJets.to_pickle(saveTo['GenJets']+tag+'.pkl')

        # store tags to file
        tagsFile.write(tag+'\n')

    tagsFile.close()

    print('** INFO : ALL DONE!')