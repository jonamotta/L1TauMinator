#include <TLorentzVector.h>
#include <TNtuple.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
#include "L1TauMinator/DataFormats/interface/GenHelper.h"
#include "L1TauMinator/DataFormats/interface/TauHelper.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


class L1CaloTauNtuplizer : public edm::stream::EDAnalyzer<> {
    public:
        explicit L1CaloTauNtuplizer(const edm::ParameterSet&);
        virtual ~L1CaloTauNtuplizer();

    private:
        //----edm control---
        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void beginRun(edm::Run const&, edm::EventSetup const&);
        virtual void endRun(edm::Run const&, edm::EventSetup const&);
        virtual void beginJob();
        virtual void endJob();
        void Initialize();

        //----private functions----

        //----tokens and handles----
        edm::EDGetTokenT<TowerHelper::SimpleTowerClustersCollection> l1TowerClustersNxMCBToken;
        edm::Handle<TowerHelper::SimpleTowerClustersCollection> l1TowerClustersNxMCBHandle;

        edm::EDGetTokenT<TowerHelper::SimpleTowerClustersCollection> l1TowerClustersNxMCEToken;
        edm::Handle<TowerHelper::SimpleTowerClustersCollection> l1TowerClustersNxMCEHandle;

        edm::EDGetTokenT<HGClusterHelper::HGClustersCollection> HGClustersToken;
        edm::Handle<HGClusterHelper::HGClustersCollection> HGClustersHandle;

        edm::EDGetTokenT<TauHelper::TausCollection> TauMinatorTausToken;
        edm::Handle<TauHelper::TausCollection> TauMinatorTausHandle;

        edm::EDGetTokenT<l1t::TauBxCollection> squareTausToken;
        edm::Handle<BXVector<l1t::Tau>>  squareTausHandle;

        edm::EDGetTokenT<GenHelper::GenTausCollection> genTausToken;
        edm::Handle<GenHelper::GenTausCollection> genTausHandle;

        edm::EDGetTokenT<GenHelper::GenJetsCollection> genBJetsToken;
        edm::Handle<GenHelper::GenJetsCollection> genBJetsHandle;

        //----private variables----
        bool DEBUG;

        TTree *_tree;
        TTree *_triggerNamesTree;
        std::string _treeName;

        ULong64_t _evtNumber;
        Int_t     _runNumber;

        std::vector<float> _tau_eta;
        std::vector<float> _tau_phi;
        std::vector<float> _tau_pt;
        std::vector<float> _tau_e;
        std::vector<float> _tau_m;
        std::vector<float> _tau_visEta;
        std::vector<float> _tau_visPhi;
        std::vector<float> _tau_visPt;
        std::vector<float> _tau_visE;
        std::vector<float> _tau_visM;
        std::vector<float> _tau_visPtEm;
        std::vector<float> _tau_visPtHad;
        std::vector<float> _tau_visEEm;
        std::vector<float> _tau_visEHad;
        std::vector<int>   _tau_DM;
        std::vector<int>   _tau_Idx;

        std::vector<float> _bjet_eta;
        std::vector<float> _bjet_phi;
        std::vector<float> _bjet_pt;
        std::vector<float> _bjet_e;

        std::vector<float>  _minatedl1tau_pt;
        std::vector<float>  _minatedl1tau_eta;
        std::vector<float>  _minatedl1tau_phi;
        std::vector<float>  _minatedl1tau_IDscore;

        std::vector<float>  _squarel1tau_pt;
        std::vector<float>  _squarel1tau_eta;
        std::vector<float>  _squarel1tau_phi;
        std::vector<float>  _squarel1tau_isoEt;
        std::vector<int>    _squarel1tau_qual;
        std::vector<int>    _squarel1tau_iso;

        std::vector<float> _cl3d_calibPt;
        std::vector<float> _cl3d_IDscore;
        std::vector<float> _cl3d_pt;
        std::vector<float> _cl3d_energy;
        std::vector<float> _cl3d_eta;
        std::vector<float> _cl3d_phi;
        std::vector<float> _cl3d_showerlength;
        std::vector<float> _cl3d_coreshowerlength;
        std::vector<float> _cl3d_firstlayer;
        std::vector<float> _cl3d_seetot;
        std::vector<float> _cl3d_seemax;
        std::vector<float> _cl3d_spptot;
        std::vector<float> _cl3d_sppmax;
        std::vector<float> _cl3d_szz;
        std::vector<float> _cl3d_srrtot;
        std::vector<float> _cl3d_srrmax;
        std::vector<float> _cl3d_srrmean;
        std::vector<float> _cl3d_hoe;
        std::vector<float> _cl3d_meanz;
        std::vector<int>   _cl3d_quality;
        std::vector<int>   _cl3d_tauMatchIdx;
        std::vector<int>   _cl3d_jetMatchIdx;
        
        std::vector<float> _clNxM_CB_calibPt;
        std::vector<float> _clNxM_CB_IDscore;
        std::vector<bool>  _clNxM_CB_barrelSeeded;
        std::vector<int>   _clNxM_CB_nEGs;
        std::vector<int>   _clNxM_CB_seedIeta;
        std::vector<int>   _clNxM_CB_seedIphi;
        std::vector<float> _clNxM_CB_seedEta;
        std::vector<float> _clNxM_CB_seedPhi;
        std::vector< std::vector<float> > _clNxM_CB_towerEta;
        std::vector< std::vector<float> > _clNxM_CB_towerPhi;
        std::vector< std::vector<float> > _clNxM_CB_towerEm;
        std::vector< std::vector<float> > _clNxM_CB_towerHad;
        std::vector< std::vector<float> > _clNxM_CB_towerEt;
        std::vector< std::vector<float> > _clNxM_CB_towerEgEt;
        std::vector< std::vector<int> >   _clNxM_CB_towerIeta;
        std::vector< std::vector<int> >   _clNxM_CB_towerIphi;
        std::vector< std::vector<int> >   _clNxM_CB_towerNeg;

        std::vector<float> _clNxM_CE_calibPt;
        std::vector<float> _clNxM_CE_IDscore;
        std::vector<bool>  _clNxM_CE_barrelSeeded;
        std::vector<int>   _clNxM_CE_nEGs;
        std::vector<int>   _clNxM_CE_seedIeta;
        std::vector<int>   _clNxM_CE_seedIphi;
        std::vector<float> _clNxM_CE_seedEta;
        std::vector<float> _clNxM_CE_seedPhi;
        std::vector< std::vector<float> > _clNxM_CE_towerEta;
        std::vector< std::vector<float> > _clNxM_CE_towerPhi;
        std::vector< std::vector<float> > _clNxM_CE_towerEm;
        std::vector< std::vector<float> > _clNxM_CE_towerHad;
        std::vector< std::vector<float> > _clNxM_CE_towerEt;
        std::vector< std::vector<float> > _clNxM_CE_towerEgEt;
        std::vector< std::vector<int> >   _clNxM_CE_towerIeta;
        std::vector< std::vector<int> >   _clNxM_CE_towerIphi;
        std::vector< std::vector<int> >   _clNxM_CE_towerNeg;

        bool brokenTensorflowPrediction = false;
};

/*
██ ███    ███ ██████  ██      ███████ ███    ███ ███████ ███    ██ ████████  █████  ████████ ██  ██████  ███    ██
██ ████  ████ ██   ██ ██      ██      ████  ████ ██      ████   ██    ██    ██   ██    ██    ██ ██    ██ ████   ██
██ ██ ████ ██ ██████  ██      █████   ██ ████ ██ █████   ██ ██  ██    ██    ███████    ██    ██ ██    ██ ██ ██  ██
██ ██  ██  ██ ██      ██      ██      ██  ██  ██ ██      ██  ██ ██    ██    ██   ██    ██    ██ ██    ██ ██  ██ ██
██ ██      ██ ██      ███████ ███████ ██      ██ ███████ ██   ████    ██    ██   ██    ██    ██  ██████  ██   ████
*/

// ----Constructor and Destructor -----
L1CaloTauNtuplizer::L1CaloTauNtuplizer(const edm::ParameterSet& iConfig)
    : l1TowerClustersNxMCBToken(consumes<TowerHelper::SimpleTowerClustersCollection>(iConfig.getParameter<edm::InputTag>("l1TowerClustersNxMCB"))),
      l1TowerClustersNxMCEToken(consumes<TowerHelper::SimpleTowerClustersCollection>(iConfig.getParameter<edm::InputTag>("l1TowerClustersNxMCE"))),
      HGClustersToken(consumes<HGClusterHelper::HGClustersCollection>(iConfig.getParameter<edm::InputTag>("HGClusters"))),
      TauMinatorTausToken(consumes<TauHelper::TausCollection>(iConfig.getParameter<edm::InputTag>("TauMinatorTaus"))),
      squareTausToken(consumes<l1t::TauBxCollection>(iConfig.getParameter<edm::InputTag>("squareTaus"))),
      genTausToken(consumes<GenHelper::GenTausCollection>(iConfig.getParameter<edm::InputTag>("genTaus"))),
      genBJetsToken(consumes<GenHelper::GenJetsCollection>(iConfig.getParameter<edm::InputTag>("genBJets"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{
    _treeName = iConfig.getParameter<std::string>("treeName");
    this -> Initialize();
    return;
}

L1CaloTauNtuplizer::~L1CaloTauNtuplizer() {}

void L1CaloTauNtuplizer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void L1CaloTauNtuplizer::Initialize()
{
    _evtNumber = 0;
    _runNumber = 0;

    _tau_eta.clear();
    _tau_phi.clear();
    _tau_pt.clear();
    _tau_e.clear();
    _tau_m.clear();
    _tau_visEta.clear();
    _tau_visPhi.clear();
    _tau_visPt.clear();
    _tau_visE.clear();
    _tau_visM.clear();
    _tau_visPtEm.clear();
    _tau_visPtHad.clear();
    _tau_visEEm.clear();
    _tau_visEHad.clear();
    _tau_DM.clear();
    _tau_Idx.clear();

    _bjet_eta.clear();
    _bjet_phi.clear();
    _bjet_pt.clear();
    _bjet_e.clear();

    _minatedl1tau_pt.clear();
    _minatedl1tau_eta.clear();
    _minatedl1tau_phi.clear();
    _minatedl1tau_IDscore.clear();

    _squarel1tau_pt.clear();
    _squarel1tau_eta.clear();
    _squarel1tau_phi.clear();
    _squarel1tau_isoEt.clear();
    _squarel1tau_qual.clear();
    _squarel1tau_iso.clear();

    _cl3d_calibPt.clear();
    _cl3d_IDscore.clear();
    _cl3d_pt.clear();
    _cl3d_energy.clear();
    _cl3d_eta.clear();
    _cl3d_phi.clear();
    _cl3d_showerlength.clear();
    _cl3d_coreshowerlength.clear();
    _cl3d_firstlayer.clear();
    _cl3d_seetot.clear();
    _cl3d_seemax.clear();
    _cl3d_spptot.clear();
    _cl3d_sppmax.clear();
    _cl3d_szz.clear();
    _cl3d_srrtot.clear();
    _cl3d_srrmax.clear();
    _cl3d_srrmean.clear();
    _cl3d_hoe.clear();
    _cl3d_meanz.clear();
    _cl3d_quality.clear();
    _cl3d_tauMatchIdx.clear();
    _cl3d_jetMatchIdx.clear();

    _clNxM_CB_calibPt.clear();
    _clNxM_CB_IDscore.clear();
    _clNxM_CB_barrelSeeded.clear();
    _clNxM_CB_nEGs.clear();
    _clNxM_CB_seedIeta.clear();
    _clNxM_CB_seedIphi.clear();
    _clNxM_CB_seedEta.clear();
    _clNxM_CB_seedPhi.clear();
    _clNxM_CB_towerEta.clear();
    _clNxM_CB_towerPhi.clear();
    _clNxM_CB_towerEm.clear();
    _clNxM_CB_towerHad.clear();
    _clNxM_CB_towerEt.clear();
    _clNxM_CB_towerEgEt.clear();
    _clNxM_CB_towerIeta.clear();
    _clNxM_CB_towerIphi.clear();
    _clNxM_CB_towerNeg.clear();

    _clNxM_CE_calibPt.clear();
    _clNxM_CE_IDscore.clear();
    _clNxM_CE_barrelSeeded.clear();
    _clNxM_CE_nEGs.clear();
    _clNxM_CE_seedIeta.clear();
    _clNxM_CE_seedIphi.clear();
    _clNxM_CE_seedEta.clear();
    _clNxM_CE_seedPhi.clear();
    _clNxM_CE_towerEta.clear();
    _clNxM_CE_towerPhi.clear();
    _clNxM_CE_towerEm.clear();
    _clNxM_CE_towerHad.clear();
    _clNxM_CE_towerEt.clear();
    _clNxM_CE_towerEgEt.clear();
    _clNxM_CE_towerIeta.clear();
    _clNxM_CE_towerIphi.clear();
    _clNxM_CE_towerNeg.clear();
}

void L1CaloTauNtuplizer::beginJob()
{
    edm::Service<TFileService> fs;
    _tree = fs -> make<TTree>(this -> _treeName.c_str(), this -> _treeName.c_str());

    _tree -> Branch("EventNumber",&_evtNumber,"EventNumber/l");
    _tree -> Branch("RunNumber",&_runNumber,"RunNumber/I");

    _tree -> Branch("tau_eta",      &_tau_eta);
    _tree -> Branch("tau_phi",      &_tau_phi);
    _tree -> Branch("tau_pt",       &_tau_pt);
    _tree -> Branch("tau_e",        &_tau_e);
    _tree -> Branch("tau_m",        &_tau_m);
    _tree -> Branch("tau_visEta",   &_tau_visEta);
    _tree -> Branch("tau_visPhi",   &_tau_visPhi);
    _tree -> Branch("tau_visPt",    &_tau_visPt);
    _tree -> Branch("tau_visE",     &_tau_visE);
    _tree -> Branch("tau_visM",     &_tau_visM);
    _tree -> Branch("tau_visPtEm",  &_tau_visPtEm);
    _tree -> Branch("tau_visPtHad", &_tau_visPtHad);
    _tree -> Branch("tau_visEEm",   &_tau_visEEm);
    _tree -> Branch("tau_visEHad",  &_tau_visEHad);
    _tree -> Branch("tau_DM",       &_tau_DM);
    _tree -> Branch("tau_Idx",      &_tau_Idx);

    _tree -> Branch("bjet_eta",     &_bjet_eta);
    _tree -> Branch("bjet_phi",     &_bjet_phi);
    _tree -> Branch("bjet_pt",      &_bjet_pt);
    _tree -> Branch("bjet_e",       &_bjet_e);

    _tree -> Branch("minatedl1tau_pt",          &_minatedl1tau_pt);
    _tree -> Branch("minatedl1tau_eta",         &_minatedl1tau_eta);
    _tree -> Branch("minatedl1tau_phi",         &_minatedl1tau_phi);
    _tree -> Branch("minatedl1tau_IDscore",     &_minatedl1tau_IDscore);

    _tree -> Branch("squarel1tau_pt",    &_squarel1tau_pt);
    _tree -> Branch("squarel1tau_eta",   &_squarel1tau_eta);
    _tree -> Branch("squarel1tau_phi",   &_squarel1tau_phi);
    _tree -> Branch("squarel1tau_isoEt", &_squarel1tau_isoEt);
    _tree -> Branch("squarel1tau_qual",  &_squarel1tau_qual);
    _tree -> Branch("squarel1tau_iso",   &_squarel1tau_iso);

    _tree -> Branch("cl3d_calibPt",          &_cl3d_calibPt);
    _tree -> Branch("cl3d_IDscore",          &_cl3d_IDscore);
    _tree -> Branch("cl3d_pt",               &_cl3d_pt);
    _tree -> Branch("cl3d_energy",           &_cl3d_energy);
    _tree -> Branch("cl3d_eta",              &_cl3d_eta);
    _tree -> Branch("cl3d_phi",              &_cl3d_phi);
    _tree -> Branch("cl3d_showerlength",     &_cl3d_showerlength);
    _tree -> Branch("cl3d_coreshowerlength", &_cl3d_coreshowerlength);
    _tree -> Branch("cl3d_firstlayer",       &_cl3d_firstlayer);
    _tree -> Branch("cl3d_seetot",           &_cl3d_seetot);
    _tree -> Branch("cl3d_seemax",           &_cl3d_seemax);
    _tree -> Branch("cl3d_spptot",           &_cl3d_spptot);
    _tree -> Branch("cl3d_sppmax",           &_cl3d_sppmax);
    _tree -> Branch("cl3d_szz",              &_cl3d_szz);
    _tree -> Branch("cl3d_srrtot",           &_cl3d_srrtot);
    _tree -> Branch("cl3d_srrmax",           &_cl3d_srrmax);
    _tree -> Branch("cl3d_srrmean",          &_cl3d_srrmean);
    _tree -> Branch("cl3d_hoe",              &_cl3d_hoe);
    _tree -> Branch("cl3d_meanz",            &_cl3d_meanz);
    _tree -> Branch("cl3d_quality",          &_cl3d_quality);
    _tree -> Branch("cl3d_tauMatchIdx",      &_cl3d_tauMatchIdx);
    _tree -> Branch("cl3d_jetMatchIdx",      &_cl3d_jetMatchIdx);

    _tree -> Branch("clNxM_CB_calibPt",         &_clNxM_CB_calibPt);
    _tree -> Branch("clNxM_CB_IDscore",         &_clNxM_CB_IDscore);
    _tree -> Branch("clNxM_CB_barrelSeeded",    &_clNxM_CB_barrelSeeded);
    _tree -> Branch("clNxM_CB_nEGs",            &_clNxM_CB_nEGs);
    _tree -> Branch("clNxM_CB_seedIeta",        &_clNxM_CB_seedIeta);
    _tree -> Branch("clNxM_CB_seedIphi",        &_clNxM_CB_seedIphi);
    _tree -> Branch("clNxM_CB_seedEta",         &_clNxM_CB_seedEta);
    _tree -> Branch("clNxM_CB_seedPhi",         &_clNxM_CB_seedPhi);
    _tree -> Branch("clNxM_CB_towerEta",        &_clNxM_CB_towerEta);
    _tree -> Branch("clNxM_CB_towerPhi",        &_clNxM_CB_towerPhi);
    _tree -> Branch("clNxM_CB_towerEm",         &_clNxM_CB_towerEm);
    _tree -> Branch("clNxM_CB_towerHad",        &_clNxM_CB_towerHad);
    _tree -> Branch("clNxM_CB_towerEt",         &_clNxM_CB_towerEt);
    _tree -> Branch("clNxM_CB_towerEgEt",       &_clNxM_CB_towerEgEt);
    _tree -> Branch("clNxM_CB_towerIeta",       &_clNxM_CB_towerIeta);
    _tree -> Branch("clNxM_CB_towerIphi",       &_clNxM_CB_towerIphi);
    _tree -> Branch("clNxM_CB_towerNeg",        &_clNxM_CB_towerNeg);

    _tree -> Branch("clNxM_CE_calibPt",         &_clNxM_CE_calibPt);
    _tree -> Branch("clNxM_CE_IDscore",         &_clNxM_CE_IDscore);
    _tree -> Branch("clNxM_CE_barrelSeeded",    &_clNxM_CE_barrelSeeded);
    _tree -> Branch("clNxM_CE_nEGs",            &_clNxM_CE_nEGs);
    _tree -> Branch("clNxM_CE_seedIeta",        &_clNxM_CE_seedIeta);
    _tree -> Branch("clNxM_CE_seedIphi",        &_clNxM_CE_seedIphi);
    _tree -> Branch("clNxM_CE_seedEta",         &_clNxM_CE_seedEta);
    _tree -> Branch("clNxM_CE_seedPhi",         &_clNxM_CE_seedPhi);
    _tree -> Branch("clNxM_CE_towerEta",        &_clNxM_CE_towerEta);
    _tree -> Branch("clNxM_CE_towerPhi",        &_clNxM_CE_towerPhi);
    _tree -> Branch("clNxM_CE_towerEm",         &_clNxM_CE_towerEm);
    _tree -> Branch("clNxM_CE_towerHad",        &_clNxM_CE_towerHad);
    _tree -> Branch("clNxM_CE_towerEt",         &_clNxM_CE_towerEt);
    _tree -> Branch("clNxM_CE_towerEgEt",       &_clNxM_CE_towerEgEt);
    _tree -> Branch("clNxM_CE_towerIeta",       &_clNxM_CE_towerIeta);
    _tree -> Branch("clNxM_CE_towerIphi",       &_clNxM_CE_towerIphi);
    _tree -> Branch("clNxM_CE_towerNeg",        &_clNxM_CE_towerNeg);

    return;
}

void L1CaloTauNtuplizer::endJob()
{ 
    if (brokenTensorflowPrediction)
    {
        std::cout << "** ERROR: the Tensorflow inference went ballistic in this job! Please re-run it!" << std::endl;
    }

    return;
}

void L1CaloTauNtuplizer::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) { return; }

void L1CaloTauNtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    this -> Initialize();

    _evtNumber = iEvent.id().event();
    _runNumber = iEvent.id().run();

    iEvent.getByToken(l1TowerClustersNxMCBToken, l1TowerClustersNxMCBHandle);
    iEvent.getByToken(l1TowerClustersNxMCEToken, l1TowerClustersNxMCEHandle);
    iEvent.getByToken(HGClustersToken, HGClustersHandle);
    iEvent.getByToken(TauMinatorTausToken, TauMinatorTausHandle);
    iEvent.getByToken(squareTausToken, squareTausHandle);
    iEvent.getByToken(genTausToken, genTausHandle);
    iEvent.getByToken(genBJetsToken, genBJetsHandle);
    
    const TowerHelper::SimpleTowerClustersCollection& l1TowerClustersNxMCB = *l1TowerClustersNxMCBHandle;
    const TowerHelper::SimpleTowerClustersCollection& l1TowerClustersNxMCE = *l1TowerClustersNxMCEHandle;
    const HGClusterHelper::HGClustersCollection& HGClusters = *HGClustersHandle;
    const TauHelper::TausCollection& TauMinatorTaus = *TauMinatorTausHandle;
    const GenHelper::GenTausCollection& genTaus = *genTausHandle;
    const GenHelper::GenJetsCollection& genBJets = *genBJetsHandle;

    if (DEBUG)
    {
        std::cout << "***************************************************************************************************************************************" << std::endl;
        std::cout << " ** total number of CB NxM clusters = " << l1TowerClustersNxMCB.size() << std::endl;
        std::cout << " ** total number of CE NxM clusters = " << l1TowerClustersNxMCE.size() << std::endl;
        std::cout << " ** total number of hgc clusters = " << HGClusters.size() << std::endl;
        std::cout << " ** total number of taus = " << TauMinatorTaus.size() << std::endl;
        std::cout << "***************************************************************************************************************************************" << std::endl;
    }

    //***************************************************************************************
    //***************************************************************************************
    // FILL TTREE 

    // Fill GenTau branches
    for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
    {
        GenHelper::GenTau tau = genTaus[tauIdx];

        _tau_eta.push_back(tau.eta);
        _tau_phi.push_back(tau.phi);
        _tau_pt.push_back(tau.pt);
        _tau_e.push_back(tau.e);
        _tau_m.push_back(tau.m);
        _tau_visEta.push_back(tau.visEta);
        _tau_visPhi.push_back(tau.visPhi);
        _tau_visPt.push_back(tau.visPt);
        _tau_visE.push_back(tau.visE);
        _tau_visM.push_back(tau.visM);
        _tau_visPtEm.push_back(tau.visPtEm);
        _tau_visPtHad.push_back(tau.visPtHad);
        _tau_visEEm.push_back(tau.visEEm);
        _tau_visEHad.push_back(tau.visEHad);
        _tau_DM.push_back(tau.DM);
        _tau_Idx.push_back(tauIdx);
    }

    // Fill GenBJet branches
    for (long unsigned int jetIdx = 0; jetIdx < genBJets.size(); jetIdx++)
    {
        GenHelper::GenJet bjet = genBJets[jetIdx];

        _bjet_eta.push_back(bjet.eta);
        _bjet_phi.push_back(bjet.phi);
        _bjet_pt.push_back(bjet.pt);
        _bjet_e.push_back(bjet.e);
    }

    // Fill TauMinator L1Taus
    for (long unsigned int tauIdx = 0; tauIdx < TauMinatorTaus.size(); tauIdx++)
    {
        TauHelper::Tau tau = TauMinatorTaus[tauIdx];

        _minatedl1tau_pt.push_back(tau.pt);
        _minatedl1tau_eta.push_back(tau.eta);
        _minatedl1tau_phi.push_back(tau.phi);
        _minatedl1tau_IDscore.push_back(tau.IDscore);

        if (tau.pt > 10000) { brokenTensorflowPrediction = true; }
    }

    // Fill baseline square L1Taus
    for (l1t::TauBxCollection::const_iterator bx0TauIt = squareTausHandle->begin(0); bx0TauIt != squareTausHandle->end(0) ; bx0TauIt++)
    {
        const l1t::Tau& tau = *bx0TauIt;

        _squarel1tau_pt.push_back(tau.pt());
        _squarel1tau_eta.push_back(tau.eta());
        _squarel1tau_phi.push_back(tau.phi());
        _squarel1tau_isoEt.push_back(tau.isoEt());
        _squarel1tau_qual.push_back(tau.hwQual());
        _squarel1tau_iso.push_back(tau.hwIso());
    }

    // Fill NxM CaloCluster branches with CB objects
    for (long unsigned int clIdx = 0; clIdx < l1TowerClustersNxMCB.size(); clIdx++)
    {
        TowerHelper::SimpleTowerCluster clNxM = l1TowerClustersNxMCB[clIdx];

        _clNxM_CB_calibPt.push_back(clNxM.calibPt);
        _clNxM_CB_IDscore.push_back(clNxM.IDscore);
        _clNxM_CB_barrelSeeded.push_back(clNxM.barrelSeeded);
        _clNxM_CB_seedIeta.push_back(clNxM.seedIeta);
        _clNxM_CB_seedIphi.push_back(clNxM.seedIphi);
        _clNxM_CB_seedEta.push_back(clNxM.seedEta);
        _clNxM_CB_seedPhi.push_back(clNxM.seedPhi);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clNxM.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clNxM.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clNxM.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clNxM.towerHits[i].towerEm);
            tmp_towerHad.push_back(clNxM.towerHits[i].towerHad);
            tmp_towerEt.push_back(clNxM.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clNxM.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clNxM.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clNxM.towerHits[i].towerIphi);
            tmp_towerNeg.push_back(clNxM.towerHits[i].nL1eg);

            nEGs += clNxM.towerHits[i].nL1eg;
        }

        _clNxM_CB_towerEta.push_back(tmp_towerEta);
        _clNxM_CB_towerPhi.push_back(tmp_towerPhi);
        _clNxM_CB_towerEm.push_back(tmp_towerEm);
        _clNxM_CB_towerHad.push_back(tmp_towerHad);
        _clNxM_CB_towerEt.push_back(tmp_towerEt);
        _clNxM_CB_towerEgEt.push_back(tmp_towerEgEt);
        _clNxM_CB_towerIeta.push_back(tmp_towerIeta);
        _clNxM_CB_towerIphi.push_back(tmp_towerIphi);
        _clNxM_CB_towerNeg.push_back(tmp_towerNeg);

        _clNxM_CB_nEGs.push_back(nEGs);
    }

    // Fill NxM CaloCluster branches with CE objects
    for (long unsigned int clIdx = 0; clIdx < l1TowerClustersNxMCE.size(); clIdx++)
    {
        TowerHelper::SimpleTowerCluster clNxM = l1TowerClustersNxMCE[clIdx];

        _clNxM_CE_calibPt.push_back(clNxM.calibPt);
        _clNxM_CE_IDscore.push_back(clNxM.IDscore);
        _clNxM_CE_barrelSeeded.push_back(clNxM.barrelSeeded);
        _clNxM_CE_seedIeta.push_back(clNxM.seedIeta);
        _clNxM_CE_seedIphi.push_back(clNxM.seedIphi);
        _clNxM_CE_seedEta.push_back(clNxM.seedEta);
        _clNxM_CE_seedPhi.push_back(clNxM.seedPhi);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clNxM.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clNxM.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clNxM.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clNxM.towerHits[i].towerEm);
            tmp_towerHad.push_back(clNxM.towerHits[i].towerHad);
            tmp_towerEt.push_back(clNxM.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clNxM.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clNxM.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clNxM.towerHits[i].towerIphi);
            tmp_towerNeg.push_back(clNxM.towerHits[i].nL1eg);

            nEGs += clNxM.towerHits[i].nL1eg;
        }

        _clNxM_CE_towerEta.push_back(tmp_towerEta);
        _clNxM_CE_towerPhi.push_back(tmp_towerPhi);
        _clNxM_CE_towerEm.push_back(tmp_towerEm);
        _clNxM_CE_towerHad.push_back(tmp_towerHad);
        _clNxM_CE_towerEt.push_back(tmp_towerEt);
        _clNxM_CE_towerEgEt.push_back(tmp_towerEgEt);
        _clNxM_CE_towerIeta.push_back(tmp_towerIeta);
        _clNxM_CE_towerIphi.push_back(tmp_towerIphi);
        _clNxM_CE_towerNeg.push_back(tmp_towerNeg);

        _clNxM_CE_nEGs.push_back(nEGs);
    }

    // Fill HGCluster branches
    for (long unsigned int hgclIdx = 0; hgclIdx < HGClusters.size(); hgclIdx++)
    {
        HGClusterHelper::HGCluster hgcl = HGClusters[hgclIdx];

        _cl3d_calibPt.push_back(hgcl.calibPt);
        _cl3d_IDscore.push_back(hgcl.IDscore);
        _cl3d_pt.push_back(hgcl.pt);
        _cl3d_energy.push_back(hgcl.energy);
        _cl3d_eta.push_back(hgcl.eta);
        _cl3d_phi.push_back(hgcl.phi);
        _cl3d_showerlength.push_back(hgcl.showerlength);
        _cl3d_coreshowerlength.push_back(hgcl.coreshowerlength);
        _cl3d_firstlayer.push_back(hgcl.firstlayer);
        _cl3d_seetot.push_back(hgcl.seetot);
        _cl3d_seemax.push_back(hgcl.seemax);
        _cl3d_spptot.push_back(hgcl.spptot);
        _cl3d_sppmax.push_back(hgcl.sppmax);
        _cl3d_szz.push_back(hgcl.szz);
        _cl3d_srrtot.push_back(hgcl.srrtot);
        _cl3d_srrmax.push_back(hgcl.srrmax);
        _cl3d_srrmean.push_back(hgcl.srrmean);
        _cl3d_hoe.push_back(hgcl.hoe);
        _cl3d_meanz.push_back(hgcl.meanz);
        _cl3d_quality.push_back(hgcl.quality);
    }

    if (DEBUG) { std::cout << " ** finished macthing, now filling the tree for run " << _runNumber << " - event " << _evtNumber << std::endl; }

    // Fill tree
    _tree -> Fill();
}

DEFINE_FWK_MODULE(L1CaloTauNtuplizer);