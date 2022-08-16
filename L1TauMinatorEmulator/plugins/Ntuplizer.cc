#include <TLorentzVector.h>
#include <TNtuple.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
// #include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
#include "L1TauMinator/DataFormats/interface/GenHelper.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


// class Ntuplizer : public edm::stream::EDAnalyzer<> {
class Ntuplizer : public edm::EDAnalyzer {
    public:
        explicit Ntuplizer(const edm::ParameterSet&);
        virtual ~Ntuplizer();

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
        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters9x9Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters9x9Handle;
        
        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters7x7Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters7x7Handle;

        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters5x5Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters5x5Handle;

        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters5x9Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters5x9Handle;

        edm::EDGetTokenT<HGClusterHelper::HGClustersCollection> HGClustersToken;
        edm::Handle<HGClusterHelper::HGClustersCollection> HGClustersHandle;

        edm::EDGetTokenT<GenHelper::GenTausCollection> genTausToken;
        edm::Handle<GenHelper::GenTausCollection> genTausHandle;

        edm::EDGetTokenT<GenHelper::GenJetsCollection> genJetsToken;
        edm::Handle<GenHelper::GenJetsCollection> genJetsHandle;

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

        std::vector<float> _jet_eta;
        std::vector<float> _jet_phi;
        std::vector<float> _jet_pt;
        std::vector<float> _jet_e;
        std::vector<float> _jet_eEm;
        std::vector<float> _jet_eHad;
        std::vector<float> _jet_eInv;

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

        std::vector<bool>  _cl9x9_barrelSeeded;
        std::vector<int>   _cl9x9_nHits;
        std::vector<int>   _cl9x9_seedIeta;
        std::vector<int>   _cl9x9_seedIphi;
        std::vector<float> _cl9x9_seedEta;
        std::vector<float> _cl9x9_seedPhi;
        std::vector<bool>  _cl9x9_isBarrel;
        std::vector<bool>  _cl9x9_isOverlap;
        std::vector<bool>  _cl9x9_isEndcap;
        std::vector<int>   _cl9x9_tauMatchIdx;
        std::vector<int>   _cl9x9_jetMatchIdx;
        std::vector<float> _cl9x9_totalEm;
        std::vector<float> _cl9x9_totalHad;
        std::vector<float> _cl9x9_totalEt;
        std::vector<float> _cl9x9_totalIem;
        std::vector<float> _cl9x9_totalIhad;
        std::vector<float> _cl9x9_totalIet;
        std::vector< std::vector<float> > _cl9x9_towerEta;
        std::vector< std::vector<float> > _cl9x9_towerPhi;
        std::vector< std::vector<float> > _cl9x9_towerEm;
        std::vector< std::vector<float> > _cl9x9_towerHad;
        std::vector< std::vector<float> > _cl9x9_towerEt;
        std::vector< std::vector<int> >   _cl9x9_towerIeta;
        std::vector< std::vector<int> >   _cl9x9_towerIphi;
        std::vector< std::vector<int> >   _cl9x9_towerIem;
        std::vector< std::vector<int> >   _cl9x9_towerIhad;
        std::vector< std::vector<int> >   _cl9x9_towerIet;

        std::vector<bool>  _cl7x7_barrelSeeded;
        std::vector<int>   _cl7x7_nHits;
        std::vector<int>   _cl7x7_seedIeta;
        std::vector<int>   _cl7x7_seedIphi;
        std::vector<float> _cl7x7_seedEta;
        std::vector<float> _cl7x7_seedPhi;
        std::vector<bool>  _cl7x7_isBarrel;
        std::vector<bool>  _cl7x7_isOverlap;
        std::vector<bool>  _cl7x7_isEndcap;
        std::vector<int>   _cl7x7_tauMatchIdx;
        std::vector<int>   _cl7x7_jetMatchIdx;
        std::vector<float> _cl7x7_totalEm;
        std::vector<float> _cl7x7_totalHad;
        std::vector<float> _cl7x7_totalEt;
        std::vector<float> _cl7x7_totalIem;
        std::vector<float> _cl7x7_totalIhad;
        std::vector<float> _cl7x7_totalIet;
        std::vector< std::vector<float> > _cl7x7_towerEta;
        std::vector< std::vector<float> > _cl7x7_towerPhi;
        std::vector< std::vector<float> > _cl7x7_towerEm;
        std::vector< std::vector<float> > _cl7x7_towerHad;
        std::vector< std::vector<float> > _cl7x7_towerEt;
        std::vector< std::vector<int> >   _cl7x7_towerIeta;
        std::vector< std::vector<int> >   _cl7x7_towerIphi;
        std::vector< std::vector<int> >   _cl7x7_towerIem;
        std::vector< std::vector<int> >   _cl7x7_towerIhad;
        std::vector< std::vector<int> >   _cl7x7_towerIet;

        std::vector<bool>  _cl5x5_barrelSeeded;
        std::vector<int>   _cl5x5_nHits;
        std::vector<int>   _cl5x5_seedIeta;
        std::vector<int>   _cl5x5_seedIphi;
        std::vector<float> _cl5x5_seedEta;
        std::vector<float> _cl5x5_seedPhi;
        std::vector<bool>  _cl5x5_isBarrel;
        std::vector<bool>  _cl5x5_isOverlap;
        std::vector<bool>  _cl5x5_isEndcap;
        std::vector<int>   _cl5x5_tauMatchIdx;
        std::vector<int>   _cl5x5_jetMatchIdx;
        std::vector<float> _cl5x5_totalEm;
        std::vector<float> _cl5x5_totalHad;
        std::vector<float> _cl5x5_totalEt;
        std::vector<float> _cl5x5_totalIem;
        std::vector<float> _cl5x5_totalIhad;
        std::vector<float> _cl5x5_totalIet;
        std::vector< std::vector<float> > _cl5x5_towerEta;
        std::vector< std::vector<float> > _cl5x5_towerPhi;
        std::vector< std::vector<float> > _cl5x5_towerEm;
        std::vector< std::vector<float> > _cl5x5_towerHad;
        std::vector< std::vector<float> > _cl5x5_towerEt;
        std::vector< std::vector<int> >   _cl5x5_towerIeta;
        std::vector< std::vector<int> >   _cl5x5_towerIphi;
        std::vector< std::vector<int> >   _cl5x5_towerIem;
        std::vector< std::vector<int> >   _cl5x5_towerIhad;
        std::vector< std::vector<int> >   _cl5x5_towerIet;

        std::vector<bool>  _cl5x9_barrelSeeded;
        std::vector<int>   _cl5x9_nHits;
        std::vector<int>   _cl5x9_seedIeta;
        std::vector<int>   _cl5x9_seedIphi;
        std::vector<float> _cl5x9_seedEta;
        std::vector<float> _cl5x9_seedPhi;
        std::vector<bool>  _cl5x9_isBarrel;
        std::vector<bool>  _cl5x9_isOverlap;
        std::vector<bool>  _cl5x9_isEndcap;
        std::vector<int>   _cl5x9_tauMatchIdx;
        std::vector<int>   _cl5x9_jetMatchIdx;
        std::vector<float> _cl5x9_totalEm;
        std::vector<float> _cl5x9_totalHad;
        std::vector<float> _cl5x9_totalEt;
        std::vector<float> _cl5x9_totalIem;
        std::vector<float> _cl5x9_totalIhad;
        std::vector<float> _cl5x9_totalIet;
        std::vector< std::vector<float> > _cl5x9_towerEta;
        std::vector< std::vector<float> > _cl5x9_towerPhi;
        std::vector< std::vector<float> > _cl5x9_towerEm;
        std::vector< std::vector<float> > _cl5x9_towerHad;
        std::vector< std::vector<float> > _cl5x9_towerEt;
        std::vector< std::vector<int> >   _cl5x9_towerIeta;
        std::vector< std::vector<int> >   _cl5x9_towerIphi;
        std::vector< std::vector<int> >   _cl5x9_towerIem;
        std::vector< std::vector<int> >   _cl5x9_towerIhad;
        std::vector< std::vector<int> >   _cl5x9_towerIet;

};

/*
██ ███    ███ ██████  ██      ███████ ███    ███ ███████ ███    ██ ████████  █████  ████████ ██  ██████  ███    ██
██ ████  ████ ██   ██ ██      ██      ████  ████ ██      ████   ██    ██    ██   ██    ██    ██ ██    ██ ████   ██
██ ██ ████ ██ ██████  ██      █████   ██ ████ ██ █████   ██ ██  ██    ██    ███████    ██    ██ ██    ██ ██ ██  ██
██ ██  ██  ██ ██      ██      ██      ██  ██  ██ ██      ██  ██ ██    ██    ██   ██    ██    ██ ██    ██ ██  ██ ██
██ ██      ██ ██      ███████ ███████ ██      ██ ███████ ██   ████    ██    ██   ██    ██    ██  ██████  ██   ████
*/

// ----Constructor and Destructor -----
Ntuplizer::Ntuplizer(const edm::ParameterSet& iConfig)
    : CaloClusters9x9Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters9x9"))),
      CaloClusters7x7Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters7x7"))),
      CaloClusters5x5Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters5x5"))),
      CaloClusters5x9Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters5x9"))),
      HGClustersToken(consumes<HGClusterHelper::HGClustersCollection>(iConfig.getParameter<edm::InputTag>("HGClusters"))),
      genTausToken(consumes<GenHelper::GenTausCollection>(iConfig.getParameter<edm::InputTag>("genTaus"))),
      genJetsToken(consumes<GenHelper::GenJetsCollection>(iConfig.getParameter<edm::InputTag>("genJets"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{
    _treeName = iConfig.getParameter<std::string>("treeName");
    this -> Initialize();
    return;
}

Ntuplizer::~Ntuplizer() {}

void Ntuplizer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void Ntuplizer::Initialize()
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

    _jet_eta.clear();
    _jet_phi.clear();
    _jet_pt.clear();
    _jet_e.clear();
    _jet_eEm.clear();
    _jet_eHad.clear();
    _jet_eInv.clear();

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

    _cl9x9_barrelSeeded.clear();
    _cl9x9_nHits.clear();
    _cl9x9_seedIeta.clear();
    _cl9x9_seedIphi.clear();
    _cl9x9_seedEta.clear();
    _cl9x9_seedPhi.clear();
    _cl9x9_isBarrel.clear();
    _cl9x9_isOverlap.clear();
    _cl9x9_isEndcap.clear();
    _cl9x9_tauMatchIdx.clear();
    _cl9x9_jetMatchIdx.clear();
    _cl9x9_totalEm.clear();
    _cl9x9_totalHad.clear();
    _cl9x9_totalEt.clear();
    _cl9x9_totalIem.clear();
    _cl9x9_totalIhad.clear();
    _cl9x9_totalIet.clear();
    _cl9x9_towerEta.clear();
    _cl9x9_towerPhi.clear();
    _cl9x9_towerEm.clear();
    _cl9x9_towerHad.clear();
    _cl9x9_towerEt.clear();
    _cl9x9_towerIeta.clear();
    _cl9x9_towerIphi.clear();
    _cl9x9_towerIem.clear();
    _cl9x9_towerIhad.clear();
    _cl9x9_towerIet.clear();

    _cl7x7_barrelSeeded.clear();
    _cl7x7_nHits.clear();
    _cl7x7_seedIeta.clear();
    _cl7x7_seedIphi.clear();
    _cl7x7_seedEta.clear();
    _cl7x7_seedPhi.clear();
    _cl7x7_isBarrel.clear();
    _cl7x7_isOverlap.clear();
    _cl7x7_isEndcap.clear();
    _cl7x7_tauMatchIdx.clear();
    _cl7x7_jetMatchIdx.clear();
    _cl7x7_totalEm.clear();
    _cl7x7_totalHad.clear();
    _cl7x7_totalEt.clear();
    _cl7x7_totalIem.clear();
    _cl7x7_totalIhad.clear();
    _cl7x7_totalIet.clear();
    _cl7x7_towerEta.clear();
    _cl7x7_towerPhi.clear();
    _cl7x7_towerEm.clear();
    _cl7x7_towerHad.clear();
    _cl7x7_towerEt.clear();
    _cl7x7_towerIeta.clear();
    _cl7x7_towerIphi.clear();
    _cl7x7_towerIem.clear();
    _cl7x7_towerIhad.clear();
    _cl7x7_towerIet.clear();

    _cl5x5_barrelSeeded.clear();
    _cl5x5_nHits.clear();
    _cl5x5_seedIeta.clear();
    _cl5x5_seedIphi.clear();
    _cl5x5_seedEta.clear();
    _cl5x5_seedPhi.clear();
    _cl5x5_isBarrel.clear();
    _cl5x5_isOverlap.clear();
    _cl5x5_isEndcap.clear();
    _cl5x5_tauMatchIdx.clear();
    _cl5x5_jetMatchIdx.clear();
    _cl5x5_totalEm.clear();
    _cl5x5_totalHad.clear();
    _cl5x5_totalEt.clear();
    _cl5x5_totalIem.clear();
    _cl5x5_totalIhad.clear();
    _cl5x5_totalIet.clear();
    _cl5x5_towerEta.clear();
    _cl5x5_towerPhi.clear();
    _cl5x5_towerEm.clear();
    _cl5x5_towerHad.clear();
    _cl5x5_towerEt.clear();
    _cl5x5_towerIeta.clear();
    _cl5x5_towerIphi.clear();
    _cl5x5_towerIem.clear();
    _cl5x5_towerIhad.clear();
    _cl5x5_towerIet.clear();

    _cl5x9_barrelSeeded.clear();
    _cl5x9_nHits.clear();
    _cl5x9_seedIeta.clear();
    _cl5x9_seedIphi.clear();
    _cl5x9_seedEta.clear();
    _cl5x9_seedPhi.clear();
    _cl5x9_isBarrel.clear();
    _cl5x9_isOverlap.clear();
    _cl5x9_isEndcap.clear();
    _cl5x9_tauMatchIdx.clear();
    _cl5x9_jetMatchIdx.clear();
    _cl5x9_totalEm.clear();
    _cl5x9_totalHad.clear();
    _cl5x9_totalEt.clear();
    _cl5x9_totalIem.clear();
    _cl5x9_totalIhad.clear();
    _cl5x9_totalIet.clear();
    _cl5x9_towerEta.clear();
    _cl5x9_towerPhi.clear();
    _cl5x9_towerEm.clear();
    _cl5x9_towerHad.clear();
    _cl5x9_towerEt.clear();
    _cl5x9_towerIeta.clear();
    _cl5x9_towerIphi.clear();
    _cl5x9_towerIem.clear();
    _cl5x9_towerIhad.clear();
    _cl5x9_towerIet.clear();
}

void Ntuplizer::beginJob()
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

    _tree -> Branch("jet_eta",  &_jet_eta);
    _tree -> Branch("jet_phi",  &_jet_phi);
    _tree -> Branch("jet_pt",   &_jet_pt);
    _tree -> Branch("jet_e",    &_jet_e);
    _tree -> Branch("jet_eEm",  &_jet_eEm);
    _tree -> Branch("jet_eHad", &_jet_eHad);
    _tree -> Branch("jet_eInv", &_jet_eInv);

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

    _tree -> Branch("cl9x9_barrelSeeded", &_cl9x9_barrelSeeded);
    _tree -> Branch("cl9x9_nHits",        &_cl9x9_nHits);
    _tree -> Branch("cl9x9_seedIeta",     &_cl9x9_seedIeta);
    _tree -> Branch("cl9x9_seedIphi",     &_cl9x9_seedIphi);
    _tree -> Branch("cl9x9_seedEta",      &_cl9x9_seedEta);
    _tree -> Branch("cl9x9_seedPhi",      &_cl9x9_seedPhi);
    _tree -> Branch("cl9x9_isBarrel",     &_cl9x9_isBarrel);
    _tree -> Branch("cl9x9_isOverlap",    &_cl9x9_isOverlap);
    _tree -> Branch("cl9x9_isEndcap",     &_cl9x9_isEndcap);
    _tree -> Branch("cl9x9_tauMatchIdx",  &_cl9x9_tauMatchIdx);
    _tree -> Branch("cl9x9_jetMatchIdx",  &_cl9x9_jetMatchIdx);
    _tree -> Branch("cl9x9_totalEm",      &_cl9x9_totalEm);
    _tree -> Branch("cl9x9_totalHad",     &_cl9x9_totalHad);
    _tree -> Branch("cl9x9_totalEt",      &_cl9x9_totalEt);
    _tree -> Branch("cl9x9_totalIem",     &_cl9x9_totalIem);
    _tree -> Branch("cl9x9_totalIhad",    &_cl9x9_totalIhad);
    _tree -> Branch("cl9x9_totalIet",     &_cl9x9_totalIet);
    _tree -> Branch("cl9x9_towerEta",     &_cl9x9_towerEta);
    _tree -> Branch("cl9x9_towerPhi",     &_cl9x9_towerPhi);
    _tree -> Branch("cl9x9_towerEm",      &_cl9x9_towerEm);
    _tree -> Branch("cl9x9_towerHad",     &_cl9x9_towerHad);
    _tree -> Branch("cl9x9_towerEt",      &_cl9x9_towerEt);
    _tree -> Branch("cl9x9_towerIeta",    &_cl9x9_towerIeta);
    _tree -> Branch("cl9x9_towerIphi",    &_cl9x9_towerIphi);
    _tree -> Branch("cl9x9_towerIem",     &_cl9x9_towerIem);
    _tree -> Branch("cl9x9_towerIhad",    &_cl9x9_towerIhad);
    _tree -> Branch("cl9x9_towerIet",     &_cl9x9_towerIet);

    _tree -> Branch("cl7x7_barrelSeeded", &_cl7x7_barrelSeeded);
    _tree -> Branch("cl7x7_nHits",        &_cl7x7_nHits);
    _tree -> Branch("cl7x7_seedIeta",     &_cl7x7_seedIeta);
    _tree -> Branch("cl7x7_seedIphi",     &_cl7x7_seedIphi);
    _tree -> Branch("cl7x7_seedEta",      &_cl7x7_seedEta);
    _tree -> Branch("cl7x7_seedPhi",      &_cl7x7_seedPhi);
    _tree -> Branch("cl7x7_isBarrel",     &_cl7x7_isBarrel);
    _tree -> Branch("cl7x7_isOverlap",    &_cl7x7_isOverlap);
    _tree -> Branch("cl7x7_isEndcap",     &_cl7x7_isEndcap);
    _tree -> Branch("cl7x7_tauMatchIdx",  &_cl7x7_tauMatchIdx);
    _tree -> Branch("cl7x7_jetMatchIdx",  &_cl7x7_jetMatchIdx);
    _tree -> Branch("cl7x7_totalEm",      &_cl7x7_totalEm);
    _tree -> Branch("cl7x7_totalHad",     &_cl7x7_totalHad);
    _tree -> Branch("cl7x7_totalEt",      &_cl7x7_totalEt);
    _tree -> Branch("cl7x7_totalIem",     &_cl7x7_totalIem);
    _tree -> Branch("cl7x7_totalIhad",    &_cl7x7_totalIhad);
    _tree -> Branch("cl7x7_totalIet",     &_cl7x7_totalIet);
    _tree -> Branch("cl7x7_towerEta",     &_cl7x7_towerEta);
    _tree -> Branch("cl7x7_towerPhi",     &_cl7x7_towerPhi);
    _tree -> Branch("cl7x7_towerEm",      &_cl7x7_towerEm);
    _tree -> Branch("cl7x7_towerHad",     &_cl7x7_towerHad);
    _tree -> Branch("cl7x7_towerEt",      &_cl7x7_towerEt);
    _tree -> Branch("cl7x7_towerIeta",    &_cl7x7_towerIeta);
    _tree -> Branch("cl7x7_towerIphi",    &_cl7x7_towerIphi);
    _tree -> Branch("cl7x7_towerIem",     &_cl7x7_towerIem);
    _tree -> Branch("cl7x7_towerIhad",    &_cl7x7_towerIhad);
    _tree -> Branch("cl7x7_towerIet",     &_cl7x7_towerIet);

    _tree -> Branch("cl5x5_barrelSeeded", &_cl5x5_barrelSeeded);
    _tree -> Branch("cl5x5_nHits",        &_cl5x5_nHits);
    _tree -> Branch("cl5x5_seedIeta",     &_cl5x5_seedIeta);
    _tree -> Branch("cl5x5_seedIphi",     &_cl5x5_seedIphi);
    _tree -> Branch("cl5x5_seedEta",      &_cl5x5_seedEta);
    _tree -> Branch("cl5x5_seedPhi",      &_cl5x5_seedPhi);
    _tree -> Branch("cl5x5_isBarrel",     &_cl5x5_isBarrel);
    _tree -> Branch("cl5x5_isOverlap",    &_cl5x5_isOverlap);
    _tree -> Branch("cl5x5_isEndcap",     &_cl5x5_isEndcap);
    _tree -> Branch("cl5x5_tauMatchIdx",  &_cl5x5_tauMatchIdx);
    _tree -> Branch("cl5x5_jetMatchIdx",  &_cl5x5_jetMatchIdx);
    _tree -> Branch("cl5x5_totalEm",      &_cl5x5_totalEm);
    _tree -> Branch("cl5x5_totalHad",     &_cl5x5_totalHad);
    _tree -> Branch("cl5x5_totalEt",      &_cl5x5_totalEt);
    _tree -> Branch("cl5x5_totalIem",     &_cl5x5_totalIem);
    _tree -> Branch("cl5x5_totalIhad",    &_cl5x5_totalIhad);
    _tree -> Branch("cl5x5_totalIet",     &_cl5x5_totalIet);
    _tree -> Branch("cl5x5_towerEta",     &_cl5x5_towerEta);
    _tree -> Branch("cl5x5_towerPhi",     &_cl5x5_towerPhi);
    _tree -> Branch("cl5x5_towerEm",      &_cl5x5_towerEm);
    _tree -> Branch("cl5x5_towerHad",     &_cl5x5_towerHad);
    _tree -> Branch("cl5x5_towerEt",      &_cl5x5_towerEt);
    _tree -> Branch("cl5x5_towerIeta",    &_cl5x5_towerIeta);
    _tree -> Branch("cl5x5_towerIphi",    &_cl5x5_towerIphi);
    _tree -> Branch("cl5x5_towerIem",     &_cl5x5_towerIem);
    _tree -> Branch("cl5x5_towerIhad",    &_cl5x5_towerIhad);
    _tree -> Branch("cl5x5_towerIet",     &_cl5x5_towerIet);

    _tree -> Branch("cl5x9_barrelSeeded", &_cl5x9_barrelSeeded);
    _tree -> Branch("cl5x9_nHits",        &_cl5x9_nHits);
    _tree -> Branch("cl5x9_seedIeta",     &_cl5x9_seedIeta);
    _tree -> Branch("cl5x9_seedIphi",     &_cl5x9_seedIphi);
    _tree -> Branch("cl5x9_seedEta",      &_cl5x9_seedEta);
    _tree -> Branch("cl5x9_seedPhi",      &_cl5x9_seedPhi);
    _tree -> Branch("cl5x9_isBarrel",     &_cl5x9_isBarrel);
    _tree -> Branch("cl5x9_isOverlap",    &_cl5x9_isOverlap);
    _tree -> Branch("cl5x9_isEndcap",     &_cl5x9_isEndcap);
    _tree -> Branch("cl5x9_tauMatchIdx",  &_cl5x9_tauMatchIdx);
    _tree -> Branch("cl5x9_jetMatchIdx",  &_cl5x9_jetMatchIdx);
    _tree -> Branch("cl5x9_totalEm",      &_cl5x9_totalEm);
    _tree -> Branch("cl5x9_totalHad",     &_cl5x9_totalHad);
    _tree -> Branch("cl5x9_totalEt",      &_cl5x9_totalEt);
    _tree -> Branch("cl5x9_totalIem",     &_cl5x9_totalIem);
    _tree -> Branch("cl5x9_totalIhad",    &_cl5x9_totalIhad);
    _tree -> Branch("cl5x9_totalIet",     &_cl5x9_totalIet);
    _tree -> Branch("cl5x9_towerEta",     &_cl5x9_towerEta);
    _tree -> Branch("cl5x9_towerPhi",     &_cl5x9_towerPhi);
    _tree -> Branch("cl5x9_towerEm",      &_cl5x9_towerEm);
    _tree -> Branch("cl5x9_towerHad",     &_cl5x9_towerHad);
    _tree -> Branch("cl5x9_towerEt",      &_cl5x9_towerEt);
    _tree -> Branch("cl5x9_towerIeta",    &_cl5x9_towerIeta);
    _tree -> Branch("cl5x9_towerIphi",    &_cl5x9_towerIphi);
    _tree -> Branch("cl5x9_towerIem",     &_cl5x9_towerIem);
    _tree -> Branch("cl5x9_towerIhad",    &_cl5x9_towerIhad);
    _tree -> Branch("cl5x9_towerIet",     &_cl5x9_towerIet);

    return;
}

void Ntuplizer::endJob() { return; }

void Ntuplizer::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) { return; }

void Ntuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    this -> Initialize();

    _evtNumber = iEvent.id().event();
    _runNumber = iEvent.id().run();

    iEvent.getByToken(CaloClusters9x9Token, CaloClusters9x9Handle);
    iEvent.getByToken(CaloClusters7x7Token, CaloClusters7x7Handle);
    iEvent.getByToken(CaloClusters5x5Token, CaloClusters5x5Handle);
    iEvent.getByToken(CaloClusters5x9Token, CaloClusters5x9Handle);
    iEvent.getByToken(HGClustersToken, HGClustersHandle);
    iEvent.getByToken(genTausToken, genTausHandle);
    iEvent.getByToken(genJetsToken, genJetsHandle);
    
    const TowerHelper::TowerClustersCollection& CaloClusters9x9 = *CaloClusters9x9Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters7x7 = *CaloClusters7x7Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters5x5 = *CaloClusters5x5Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters5x9 = *CaloClusters5x9Handle;
    const HGClusterHelper::HGClustersCollection& HGClusters = *HGClustersHandle;
    const GenHelper::GenTausCollection& genTaus = *genTausHandle;
    const GenHelper::GenJetsCollection& genJets = *genJetsHandle;

    if (DEBUG)
    {
        std::cout << "***************************************************************************************************************************************" << std::endl;
        std::cout << " ** total number of 9x9 clusters = " << CaloClusters9x9.size() << std::endl;
        std::cout << " ** total number of 7x7 clusters = " << CaloClusters7x7.size() << std::endl;
        std::cout << " ** total number of 5x5 clusters = " << CaloClusters5x5.size() << std::endl;
        std::cout << " ** total number of 5x9 clusters = " << CaloClusters5x9.size() << std::endl;
        std::cout << " ** total number of hgc clusters = " << HGClusters.size() << std::endl;
        std::cout << "***************************************************************************************************************************************" << std::endl;
    }

    // Perform geometrical matching of 9x9 CaloClusters to GenTaus and GenJets then directly fill branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters9x9.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu9x9 = CaloClusters9x9[cluIdx];

        for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
        {
            GenHelper::GenTau tau = genTaus[tauIdx];

            float dEta = clu9x9.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu9x9.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu9x9.tauMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt tau
                    // this in theory should never happen for taus, but more likely for jets
                    if (tau.visPt > genTaus[clu9x9.tauMatchIdx].visPt) { clu9x9.tauMatchIdx = tauIdx; }
                }
                else { clu9x9.tauMatchIdx = tauIdx; }
            }
        }

        for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
        {
            GenHelper::GenJet jet = genJets[jetIdx];

            float dEta = clu9x9.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu9x9.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu9x9.jetMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt jet
                    if (jet.pt > genJets[clu9x9.jetMatchIdx].pt) { clu9x9.jetMatchIdx = jetIdx; }
                }
                else { clu9x9.jetMatchIdx = jetIdx; }
            }
        }

        if (DEBUG)
        {
            if (clu9x9.tauMatchIdx != -99 || clu9x9.jetMatchIdx != -99)
            {
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                std::cout << "clu9x9 idx " << cluIdx << " eta " << clu9x9.seedEta << " phi " << clu9x9.seedPhi << std::endl;
                if (clu9x9.tauMatchIdx != -99) { std::cout << "tau match idx " << clu9x9.tauMatchIdx << " eta " << genTaus[clu9x9.tauMatchIdx].visEta << " phi " << genTaus[clu9x9.tauMatchIdx].visPhi << std::endl; }
                if (clu9x9.jetMatchIdx != -99) { std::cout << "jet match idx " << clu9x9.jetMatchIdx << " eta " << genJets[clu9x9.jetMatchIdx].eta << " phi " << genJets[clu9x9.jetMatchIdx].phi << std::endl; }
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
        }

        // Fill 9x9 CaloCluster branches
        _cl9x9_barrelSeeded.push_back(clu9x9.barrelSeeded);
        _cl9x9_nHits.push_back(clu9x9.nHits);
        _cl9x9_seedIeta.push_back(clu9x9.seedIeta);
        _cl9x9_seedIphi.push_back(clu9x9.seedIphi);
        _cl9x9_seedEta.push_back(clu9x9.seedEta);
        _cl9x9_seedPhi.push_back(clu9x9.seedPhi);
        _cl9x9_isBarrel.push_back(clu9x9.isBarrel);
        _cl9x9_isOverlap.push_back(clu9x9.isOverlap);
        _cl9x9_isEndcap.push_back(clu9x9.isEndcap);
        _cl9x9_tauMatchIdx.push_back(clu9x9.tauMatchIdx);
        _cl9x9_jetMatchIdx.push_back(clu9x9.jetMatchIdx);
        _cl9x9_totalEm.push_back(clu9x9.totalEm);
        _cl9x9_totalHad.push_back(clu9x9.totalHad);
        _cl9x9_totalEt.push_back(clu9x9.totalEt);
        _cl9x9_totalIem.push_back(clu9x9.totalIem);
        _cl9x9_totalIhad.push_back(clu9x9.totalIhad);
        _cl9x9_totalIet.push_back(clu9x9.totalIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;

        for (long unsigned int i = 0; i < clu9x9.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu9x9.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu9x9.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu9x9.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu9x9.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu9x9.towerHits[i].towerEt);
            tmp_towerIeta.push_back(clu9x9.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu9x9.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu9x9.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu9x9.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu9x9.towerHits[i].towerIet);
        }

        _cl9x9_towerEta.push_back(tmp_towerEta);
        _cl9x9_towerPhi.push_back(tmp_towerPhi);
        _cl9x9_towerEm.push_back(tmp_towerEm);
        _cl9x9_towerHad.push_back(tmp_towerHad);
        _cl9x9_towerEt.push_back(tmp_towerEt);
        _cl9x9_towerIeta.push_back(tmp_towerIeta);
        _cl9x9_towerIphi.push_back(tmp_towerIphi);
        _cl9x9_towerIem.push_back(tmp_towerIem);
        _cl9x9_towerIhad.push_back(tmp_towerIhad);
        _cl9x9_towerIet.push_back(tmp_towerIet);
    }


    // Perform geometrical matching of 7x7 CaloClusters to GenTaus and GenJets then directly fill branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters7x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu7x7 = CaloClusters7x7[cluIdx];

        for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
        {
            GenHelper::GenTau tau = genTaus[tauIdx];

            float dEta = clu7x7.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu7x7.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu7x7.tauMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt tau
                    // this in theory should never happen for taus, but more likely for jets
                    if (tau.visPt > genTaus[clu7x7.tauMatchIdx].visPt) { clu7x7.tauMatchIdx = tauIdx; }
                }
                else { clu7x7.tauMatchIdx = tauIdx; }
            }
        }

        for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
        {
            GenHelper::GenJet jet = genJets[jetIdx];

            float dEta = clu7x7.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu7x7.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu7x7.jetMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt jet
                    if (jet.pt > genJets[clu7x7.jetMatchIdx].pt) { clu7x7.jetMatchIdx = jetIdx; }
                }
                else { clu7x7.jetMatchIdx = jetIdx; }
            }
        }

        if (DEBUG)
        {
            if (clu7x7.tauMatchIdx != -99 || clu7x7.jetMatchIdx != -99)
            {
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                std::cout << "clu7x7 idx " << cluIdx << " eta " << clu7x7.seedEta << " phi " << clu7x7.seedPhi << std::endl;
                if (clu7x7.tauMatchIdx != -99) { std::cout << "tau match idx " << clu7x7.tauMatchIdx << " eta " << genTaus[clu7x7.tauMatchIdx].visEta << " phi " << genTaus[clu7x7.tauMatchIdx].visPhi << std::endl; }
                if (clu7x7.jetMatchIdx != -99) { std::cout << "jet match idx " << clu7x7.jetMatchIdx << " eta " << genJets[clu7x7.jetMatchIdx].eta << " phi " << genJets[clu7x7.jetMatchIdx].phi << std::endl; }
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
        }

        // Fill 7x7 CaloCluster branches
        _cl7x7_barrelSeeded.push_back(clu7x7.barrelSeeded);
        _cl7x7_nHits.push_back(clu7x7.nHits);
        _cl7x7_seedIeta.push_back(clu7x7.seedIeta);
        _cl7x7_seedIphi.push_back(clu7x7.seedIphi);
        _cl7x7_seedEta.push_back(clu7x7.seedEta);
        _cl7x7_seedPhi.push_back(clu7x7.seedPhi);
        _cl7x7_isBarrel.push_back(clu7x7.isBarrel);
        _cl7x7_isOverlap.push_back(clu7x7.isOverlap);
        _cl7x7_isEndcap.push_back(clu7x7.isEndcap);
        _cl7x7_tauMatchIdx.push_back(clu7x7.tauMatchIdx);
        _cl7x7_jetMatchIdx.push_back(clu7x7.jetMatchIdx);
        _cl7x7_totalEm.push_back(clu7x7.totalEm);
        _cl7x7_totalHad.push_back(clu7x7.totalHad);
        _cl7x7_totalEt.push_back(clu7x7.totalEt);
        _cl7x7_totalIem.push_back(clu7x7.totalIem);
        _cl7x7_totalIhad.push_back(clu7x7.totalIhad);
        _cl7x7_totalIet.push_back(clu7x7.totalIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;

        for (long unsigned int i = 0; i < clu7x7.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu7x7.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu7x7.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu7x7.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu7x7.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu7x7.towerHits[i].towerEt);
            tmp_towerIeta.push_back(clu7x7.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu7x7.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu7x7.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu7x7.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu7x7.towerHits[i].towerIet);
        }

        _cl7x7_towerEta.push_back(tmp_towerEta);
        _cl7x7_towerPhi.push_back(tmp_towerPhi);
        _cl7x7_towerEm.push_back(tmp_towerEm);
        _cl7x7_towerHad.push_back(tmp_towerHad);
        _cl7x7_towerEt.push_back(tmp_towerEt);
        _cl7x7_towerIeta.push_back(tmp_towerIeta);
        _cl7x7_towerIphi.push_back(tmp_towerIphi);
        _cl7x7_towerIem.push_back(tmp_towerIem);
        _cl7x7_towerIhad.push_back(tmp_towerIhad);
        _cl7x7_towerIet.push_back(tmp_towerIet);
    }


    // Perform geometrical matching of 5x5 CaloClusters to GenTaus and GenJets then directly fill branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x5.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x5 = CaloClusters5x5[cluIdx];

        for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
        {
            GenHelper::GenTau tau = genTaus[tauIdx];

            float dEta = clu5x5.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu5x5.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu5x5.tauMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt tau
                    // this in theory should never happen for taus, but more likely for jets
                    if (tau.visPt > genTaus[clu5x5.tauMatchIdx].visPt) { clu5x5.tauMatchIdx = tauIdx; }
                }
                else { clu5x5.tauMatchIdx = tauIdx; }
            }
        }

        for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
        {
            GenHelper::GenJet jet = genJets[jetIdx];

            float dEta = clu5x5.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu5x5.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu5x5.jetMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt jet
                    if (jet.pt > genJets[clu5x5.jetMatchIdx].pt) { clu5x5.jetMatchIdx = jetIdx; }
                }
                else { clu5x5.jetMatchIdx = jetIdx; }
            }
        }

        if (DEBUG)
        {
            if (clu5x5.tauMatchIdx != -99 || clu5x5.jetMatchIdx != -99)
            {
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                std::cout << "clu5x5 idx " << cluIdx << " eta " << clu5x5.seedEta << " phi " << clu5x5.seedPhi << std::endl;
                if (clu5x5.tauMatchIdx != -99) { std::cout << "tau match idx " << clu5x5.tauMatchIdx << " eta " << genTaus[clu5x5.tauMatchIdx].visEta << " phi " << genTaus[clu5x5.tauMatchIdx].visPhi << std::endl; }
                if (clu5x5.jetMatchIdx != -99) { std::cout << "jet match idx " << clu5x5.jetMatchIdx << " eta " << genJets[clu5x5.jetMatchIdx].eta << " phi " << genJets[clu5x5.jetMatchIdx].phi << std::endl; }
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
        }

        // Fill 5x5 CaloCluster branches
        _cl5x5_barrelSeeded.push_back(clu5x5.barrelSeeded);
        _cl5x5_nHits.push_back(clu5x5.nHits);
        _cl5x5_seedIeta.push_back(clu5x5.seedIeta);
        _cl5x5_seedIphi.push_back(clu5x5.seedIphi);
        _cl5x5_seedEta.push_back(clu5x5.seedEta);
        _cl5x5_seedPhi.push_back(clu5x5.seedPhi);
        _cl5x5_isBarrel.push_back(clu5x5.isBarrel);
        _cl5x5_isOverlap.push_back(clu5x5.isOverlap);
        _cl5x5_isEndcap.push_back(clu5x5.isEndcap);
        _cl5x5_tauMatchIdx.push_back(clu5x5.tauMatchIdx);
        _cl5x5_jetMatchIdx.push_back(clu5x5.jetMatchIdx);
        _cl5x5_totalEm.push_back(clu5x5.totalEm);
        _cl5x5_totalHad.push_back(clu5x5.totalHad);
        _cl5x5_totalEt.push_back(clu5x5.totalEt);
        _cl5x5_totalIem.push_back(clu5x5.totalIem);
        _cl5x5_totalIhad.push_back(clu5x5.totalIhad);
        _cl5x5_totalIet.push_back(clu5x5.totalIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;

        for (long unsigned int i = 0; i < clu5x5.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu5x5.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu5x5.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu5x5.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu5x5.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu5x5.towerHits[i].towerEt);
            tmp_towerIeta.push_back(clu5x5.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu5x5.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu5x5.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu5x5.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu5x5.towerHits[i].towerIet);
        }

        _cl5x5_towerEta.push_back(tmp_towerEta);
        _cl5x5_towerPhi.push_back(tmp_towerPhi);
        _cl5x5_towerEm.push_back(tmp_towerEm);
        _cl5x5_towerHad.push_back(tmp_towerHad);
        _cl5x5_towerEt.push_back(tmp_towerEt);
        _cl5x5_towerIeta.push_back(tmp_towerIeta);
        _cl5x5_towerIphi.push_back(tmp_towerIphi);
        _cl5x5_towerIem.push_back(tmp_towerIem);
        _cl5x5_towerIhad.push_back(tmp_towerIhad);
        _cl5x5_towerIet.push_back(tmp_towerIet);
    }


    // Perform geometrical matching of 5x9 CaloClusters to GenTaus and GenJets then directly fill branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x9.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x9 = CaloClusters5x9[cluIdx];

        for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
        {
            GenHelper::GenTau tau = genTaus[tauIdx];

            float dEta = clu5x9.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu5x9.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu5x9.tauMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt tau
                    // this in theory should never happen for taus, but more likely for jets
                    if (tau.visPt > genTaus[clu5x9.tauMatchIdx].visPt) { clu5x9.tauMatchIdx = tauIdx; }
                }
                else { clu5x9.tauMatchIdx = tauIdx; }
            }
        }

        for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
        {
            GenHelper::GenJet jet = genJets[jetIdx];

            float dEta = clu5x9.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu5x9.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (clu5x9.jetMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt jet
                    if (jet.pt > genJets[clu5x9.jetMatchIdx].pt) { clu5x9.jetMatchIdx = jetIdx; }
                }
                else { clu5x9.jetMatchIdx = jetIdx; }
            }
        }

        if (DEBUG)
        {
            if (clu5x9.tauMatchIdx != -99 || clu5x9.jetMatchIdx != -99)
            {
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                std::cout << "clu5x9 idx " << cluIdx << " eta " << clu5x9.seedEta << " phi " << clu5x9.seedPhi << std::endl;
                if (clu5x9.tauMatchIdx != -99) { std::cout << "tau match idx " << clu5x9.tauMatchIdx << " eta " << genTaus[clu5x9.tauMatchIdx].visEta << " phi " << genTaus[clu5x9.tauMatchIdx].visPhi << std::endl; }
                if (clu5x9.jetMatchIdx != -99) { std::cout << "jet match idx " << clu5x9.jetMatchIdx << " eta " << genJets[clu5x9.jetMatchIdx].eta << " phi " << genJets[clu5x9.jetMatchIdx].phi << std::endl; }
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
        }

        // Fill 5x9 CaloCluster branches
        _cl5x9_barrelSeeded.push_back(clu5x9.barrelSeeded);
        _cl5x9_nHits.push_back(clu5x9.nHits);
        _cl5x9_seedIeta.push_back(clu5x9.seedIeta);
        _cl5x9_seedIphi.push_back(clu5x9.seedIphi);
        _cl5x9_seedEta.push_back(clu5x9.seedEta);
        _cl5x9_seedPhi.push_back(clu5x9.seedPhi);
        _cl5x9_isBarrel.push_back(clu5x9.isBarrel);
        _cl5x9_isOverlap.push_back(clu5x9.isOverlap);
        _cl5x9_isEndcap.push_back(clu5x9.isEndcap);
        _cl5x9_tauMatchIdx.push_back(clu5x9.tauMatchIdx);
        _cl5x9_jetMatchIdx.push_back(clu5x9.jetMatchIdx);
        _cl5x9_totalEm.push_back(clu5x9.totalEm);
        _cl5x9_totalHad.push_back(clu5x9.totalHad);
        _cl5x9_totalEt.push_back(clu5x9.totalEt);
        _cl5x9_totalIem.push_back(clu5x9.totalIem);
        _cl5x9_totalIhad.push_back(clu5x9.totalIhad);
        _cl5x9_totalIet.push_back(clu5x9.totalIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;

        for (long unsigned int i = 0; i < clu5x9.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu5x9.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu5x9.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu5x9.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu5x9.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu5x9.towerHits[i].towerEt);
            tmp_towerIeta.push_back(clu5x9.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu5x9.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu5x9.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu5x9.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu5x9.towerHits[i].towerIet);
        }

        _cl5x9_towerEta.push_back(tmp_towerEta);
        _cl5x9_towerPhi.push_back(tmp_towerPhi);
        _cl5x9_towerEm.push_back(tmp_towerEm);
        _cl5x9_towerHad.push_back(tmp_towerHad);
        _cl5x9_towerEt.push_back(tmp_towerEt);
        _cl5x9_towerIeta.push_back(tmp_towerIeta);
        _cl5x9_towerIphi.push_back(tmp_towerIphi);
        _cl5x9_towerIem.push_back(tmp_towerIem);
        _cl5x9_towerIhad.push_back(tmp_towerIhad);
        _cl5x9_towerIet.push_back(tmp_towerIet);
    }


    // Perform geometrical matching of HGClusters to GenTaus and GenJets then directly fill branches
    for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
    {
        HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

        for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
        {
            GenHelper::GenTau tau = genTaus[tauIdx];

            float dEta = hgclu.eta - tau.visEta;
            float dPhi = reco::deltaPhi(hgclu.phi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (hgclu.tauMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt tau
                    // this in theory should never happen for taus, but more likely for jets
                    if (tau.visPt > genTaus[hgclu.tauMatchIdx].visPt) { hgclu.tauMatchIdx = tauIdx; }
                }
                else { hgclu.tauMatchIdx = tauIdx; }
            }

            // Fill GenTau branches
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
        }

        for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
        {
            GenHelper::GenJet jet = genJets[jetIdx];

            float dEta = hgclu.eta - jet.eta;
            float dPhi = reco::deltaPhi(hgclu.phi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25)
            {
                if (hgclu.jetMatchIdx != -99) 
                {
                    // if there has already been a match we keep the match with the highest pt jet
                    if (jet.pt > genJets[hgclu.jetMatchIdx].pt) { hgclu.jetMatchIdx = jetIdx; }
                }
                else { hgclu.jetMatchIdx = jetIdx; }
            }

            // Fill GenJet branches
            _jet_eta.push_back(jet.eta);
            _jet_phi.push_back(jet.phi);
            _jet_pt.push_back(jet.pt);
            _jet_e.push_back(jet.e);
            _jet_eEm.push_back(jet.eEm);
            _jet_eHad.push_back(jet.eHad);
            _jet_eInv.push_back(jet.eInv);
        }
    
        if (DEBUG)
        {
            if (hgclu.tauMatchIdx != -99 || hgclu.jetMatchIdx != -99)
            {
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
                std::cout << "hgclu idx " << hgcluIdx << " eta " << hgclu.eta << " phi " << hgclu.phi << std::endl;
                if (hgclu.tauMatchIdx != -99) { std::cout << "tau match idx " << hgclu.tauMatchIdx << " eta " << genTaus[hgclu.tauMatchIdx].visEta << " phi " << genTaus[hgclu.tauMatchIdx].visPhi << std::endl; }
                if (hgclu.jetMatchIdx != -99) { std::cout << "jet match idx " << hgclu.jetMatchIdx << " eta " << genJets[hgclu.jetMatchIdx].eta << " phi " << genJets[hgclu.jetMatchIdx].phi << std::endl; }
                std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }
        }

        // Fill HGCluster branches
        _cl3d_pt.push_back(hgclu.pt);
        _cl3d_energy.push_back(hgclu.energy);
        _cl3d_eta.push_back(hgclu.eta);
        _cl3d_phi.push_back(hgclu.phi);
        _cl3d_showerlength.push_back(hgclu.showerlength);
        _cl3d_coreshowerlength.push_back(hgclu.coreshowerlength);
        _cl3d_firstlayer.push_back(hgclu.firstlayer);
        _cl3d_seetot.push_back(hgclu.seetot);
        _cl3d_seemax.push_back(hgclu.seemax);
        _cl3d_spptot.push_back(hgclu.spptot);
        _cl3d_sppmax.push_back(hgclu.sppmax);
        _cl3d_szz.push_back(hgclu.szz);
        _cl3d_srrtot.push_back(hgclu.srrtot);
        _cl3d_srrmax.push_back(hgclu.srrmax);
        _cl3d_srrmean.push_back(hgclu.srrmean);
        _cl3d_hoe.push_back(hgclu.hoe);
        _cl3d_meanz.push_back(hgclu.meanz);
        _cl3d_quality.push_back(hgclu.quality);
        _cl3d_tauMatchIdx.push_back(hgclu.tauMatchIdx);
        _cl3d_jetMatchIdx.push_back(hgclu.jetMatchIdx);
    }

    if (DEBUG) { std::cout << " ** finished macthing, now filling the tree for run " << _runNumber << " - event " << _evtNumber << std::endl; }

    // Fill tree
    _tree -> Fill();
}

DEFINE_FWK_MODULE(Ntuplizer);