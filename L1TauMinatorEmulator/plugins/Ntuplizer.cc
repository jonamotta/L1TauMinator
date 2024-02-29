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

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
#include "L1TauMinator/DataFormats/interface/GenHelper.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


class Ntuplizer : public edm::stream::EDAnalyzer<> {
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

        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters5x7Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters5x7Handle;

        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters3x7Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters3x7Handle;

        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClusters3x5Token;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClusters3x5Handle;

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
        std::vector<int>   _tau_Idx;

        std::vector<float> _jet_eta;
        std::vector<float> _jet_phi;
        std::vector<float> _jet_pt;
        std::vector<float> _jet_e;
        std::vector<float> _jet_eEm;
        std::vector<float> _jet_eHad;
        std::vector<float> _jet_eInv;
        std::vector<int>   _jet_Idx;

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
        std::vector<int>   _cl3d_puid;
        std::vector<float> _cl3d_puidscore;
        std::vector<int>   _cl3d_pionid;
        std::vector<float> _cl3d_pionidscore;
        std::vector<int>   _cl3d_tauMatchIdx;
        std::vector<int>   _cl3d_jetMatchIdx;

        std::vector<bool>  _cl9x9_barrelSeeded;
        std::vector<int>   _cl9x9_nHits;
        std::vector<int>   _cl9x9_nEGs;
        std::vector<int>   _cl9x9_seedIeta;
        std::vector<int>   _cl9x9_seedIphi;
        std::vector<float> _cl9x9_seedEta;
        std::vector<float> _cl9x9_seedPhi;
        std::vector<bool>  _cl9x9_isBarrel;
        std::vector<bool>  _cl9x9_isOverlap;
        std::vector<bool>  _cl9x9_isEndcap;
        std::vector<bool>  _cl9x9_isPhiFlipped;
        std::vector<int>   _cl9x9_tauMatchIdx;
        std::vector<int>   _cl9x9_jetMatchIdx;
        std::vector<int>   _cl9x9_cl3dMatchIdx;
        std::vector<float> _cl9x9_totalEm;
        std::vector<float> _cl9x9_totalHad;
        std::vector<float> _cl9x9_totalEt;
        std::vector<float> _cl9x9_totalEgEt;
        std::vector<int> _cl9x9_totalIem;
        std::vector<int> _cl9x9_totalIhad;
        std::vector<int> _cl9x9_totalIet;
        std::vector<int> _cl9x9_totalEgIet;
        std::vector< std::vector<float> > _cl9x9_towerEta;
        std::vector< std::vector<float> > _cl9x9_towerPhi;
        std::vector< std::vector<float> > _cl9x9_towerEm;
        std::vector< std::vector<float> > _cl9x9_towerHad;
        std::vector< std::vector<float> > _cl9x9_towerEt;
        std::vector< std::vector<float> > _cl9x9_towerEgEt;
        std::vector< std::vector<int> >   _cl9x9_towerIeta;
        std::vector< std::vector<int> >   _cl9x9_towerIphi;
        std::vector< std::vector<int> >   _cl9x9_towerIem;
        std::vector< std::vector<int> >   _cl9x9_towerIhad;
        std::vector< std::vector<int> >   _cl9x9_towerIet;
        std::vector< std::vector<int> >   _cl9x9_towerEgIet;
        std::vector< std::vector<int> >   _cl9x9_towerNeg;

        std::vector<bool>  _cl7x7_barrelSeeded;
        std::vector<int>   _cl7x7_nHits;
        std::vector<int>   _cl7x7_nEGs;
        std::vector<int>   _cl7x7_seedIeta;
        std::vector<int>   _cl7x7_seedIphi;
        std::vector<float> _cl7x7_seedEta;
        std::vector<float> _cl7x7_seedPhi;
        std::vector<bool>  _cl7x7_isBarrel;
        std::vector<bool>  _cl7x7_isOverlap;
        std::vector<bool>  _cl7x7_isEndcap;
        std::vector<bool>  _cl7x7_isPhiFlipped;
        std::vector<int>   _cl7x7_tauMatchIdx;
        std::vector<int>   _cl7x7_jetMatchIdx;
        std::vector<int>   _cl7x7_cl3dMatchIdx;
        std::vector<float> _cl7x7_totalEm;
        std::vector<float> _cl7x7_totalHad;
        std::vector<float> _cl7x7_totalEt;
        std::vector<float> _cl7x7_totalEgEt;
        std::vector<int> _cl7x7_totalIem;
        std::vector<int> _cl7x7_totalIhad;
        std::vector<int> _cl7x7_totalIet;
        std::vector<int> _cl7x7_totalEgIet;
        std::vector< std::vector<float> > _cl7x7_towerEta;
        std::vector< std::vector<float> > _cl7x7_towerPhi;
        std::vector< std::vector<float> > _cl7x7_towerEm;
        std::vector< std::vector<float> > _cl7x7_towerHad;
        std::vector< std::vector<float> > _cl7x7_towerEt;
        std::vector< std::vector<float> > _cl7x7_towerEgEt;
        std::vector< std::vector<int> >   _cl7x7_towerIeta;
        std::vector< std::vector<int> >   _cl7x7_towerIphi;
        std::vector< std::vector<int> >   _cl7x7_towerIem;
        std::vector< std::vector<int> >   _cl7x7_towerIhad;
        std::vector< std::vector<int> >   _cl7x7_towerIet;
        std::vector< std::vector<int> >   _cl7x7_towerEgIet;
        std::vector< std::vector<int> >   _cl7x7_towerNeg;

        std::vector<bool>  _cl5x5_barrelSeeded;
        std::vector<int>   _cl5x5_nHits;
        std::vector<int>   _cl5x5_nEGs;
        std::vector<int>   _cl5x5_seedIeta;
        std::vector<int>   _cl5x5_seedIphi;
        std::vector<float> _cl5x5_seedEta;
        std::vector<float> _cl5x5_seedPhi;
        std::vector<bool>  _cl5x5_isBarrel;
        std::vector<bool>  _cl5x5_isOverlap;
        std::vector<bool>  _cl5x5_isEndcap;
        std::vector<bool>  _cl5x5_isPhiFlipped;
        std::vector<int>   _cl5x5_tauMatchIdx;
        std::vector<int>   _cl5x5_jetMatchIdx;
        std::vector<int>   _cl5x5_cl3dMatchIdx;
        std::vector<float> _cl5x5_totalEm;
        std::vector<float> _cl5x5_totalHad;
        std::vector<float> _cl5x5_totalEt;
        std::vector<float> _cl5x5_totalEgEt;
        std::vector<int> _cl5x5_totalIem;
        std::vector<int> _cl5x5_totalIhad;
        std::vector<int> _cl5x5_totalIet;
        std::vector<int> _cl5x5_totalEgIet;
        std::vector< std::vector<float> > _cl5x5_towerEta;
        std::vector< std::vector<float> > _cl5x5_towerPhi;
        std::vector< std::vector<float> > _cl5x5_towerEm;
        std::vector< std::vector<float> > _cl5x5_towerHad;
        std::vector< std::vector<float> > _cl5x5_towerEt;
        std::vector< std::vector<float> > _cl5x5_towerEgEt;
        std::vector< std::vector<int> >   _cl5x5_towerIeta;
        std::vector< std::vector<int> >   _cl5x5_towerIphi;
        std::vector< std::vector<int> >   _cl5x5_towerIem;
        std::vector< std::vector<int> >   _cl5x5_towerIhad;
        std::vector< std::vector<int> >   _cl5x5_towerIet;
        std::vector< std::vector<int> >   _cl5x5_towerEgIet;
        std::vector< std::vector<int> >   _cl5x5_towerNeg;

        std::vector<bool>  _cl5x9_barrelSeeded;
        std::vector<int>   _cl5x9_nHits;
        std::vector<int>   _cl5x9_nEGs;
        std::vector<int>   _cl5x9_seedIeta;
        std::vector<int>   _cl5x9_seedIphi;
        std::vector<float> _cl5x9_seedEta;
        std::vector<float> _cl5x9_seedPhi;
        std::vector<bool>  _cl5x9_isBarrel;
        std::vector<bool>  _cl5x9_isOverlap;
        std::vector<bool>  _cl5x9_isEndcap;
        std::vector<bool>  _cl5x9_isPhiFlipped;
        std::vector<int>   _cl5x9_tauMatchIdx;
        std::vector<int>   _cl5x9_jetMatchIdx;
        std::vector<int>   _cl5x9_cl3dMatchIdx;
        std::vector<float> _cl5x9_totalEm;
        std::vector<float> _cl5x9_totalHad;
        std::vector<float> _cl5x9_totalEt;
        std::vector<float> _cl5x9_totalEgEt;
        std::vector<int> _cl5x9_totalIem;
        std::vector<int> _cl5x9_totalIhad;
        std::vector<int> _cl5x9_totalIet;
        std::vector<int> _cl5x9_totalEgIet;
        std::vector< std::vector<float> > _cl5x9_towerEta;
        std::vector< std::vector<float> > _cl5x9_towerPhi;
        std::vector< std::vector<float> > _cl5x9_towerEm;
        std::vector< std::vector<float> > _cl5x9_towerHad;
        std::vector< std::vector<float> > _cl5x9_towerEt;
        std::vector< std::vector<float> > _cl5x9_towerEgEt;
        std::vector< std::vector<int> >   _cl5x9_towerIeta;
        std::vector< std::vector<int> >   _cl5x9_towerIphi;
        std::vector< std::vector<int> >   _cl5x9_towerIem;
        std::vector< std::vector<int> >   _cl5x9_towerIhad;
        std::vector< std::vector<int> >   _cl5x9_towerIet;
        std::vector< std::vector<int> >   _cl5x9_towerEgIet;
        std::vector< std::vector<int> >   _cl5x9_towerNeg;

        std::vector<bool>  _cl5x7_barrelSeeded;
        std::vector<int>   _cl5x7_nHits;
        std::vector<int>   _cl5x7_nEGs;
        std::vector<int>   _cl5x7_seedIeta;
        std::vector<int>   _cl5x7_seedIphi;
        std::vector<float> _cl5x7_seedEta;
        std::vector<float> _cl5x7_seedPhi;
        std::vector<bool>  _cl5x7_isBarrel;
        std::vector<bool>  _cl5x7_isOverlap;
        std::vector<bool>  _cl5x7_isEndcap;
        std::vector<bool>  _cl5x7_isPhiFlipped;
        std::vector<int>   _cl5x7_tauMatchIdx;
        std::vector<int>   _cl5x7_jetMatchIdx;
        std::vector<int>   _cl5x7_cl3dMatchIdx;
        std::vector<float> _cl5x7_totalEm;
        std::vector<float> _cl5x7_totalHad;
        std::vector<float> _cl5x7_totalEt;
        std::vector<float> _cl5x7_totalEgEt;
        std::vector<int> _cl5x7_totalIem;
        std::vector<int> _cl5x7_totalIhad;
        std::vector<int> _cl5x7_totalIet;
        std::vector<int> _cl5x7_totalEgIet;
        std::vector< std::vector<float> > _cl5x7_towerEta;
        std::vector< std::vector<float> > _cl5x7_towerPhi;
        std::vector< std::vector<float> > _cl5x7_towerEm;
        std::vector< std::vector<float> > _cl5x7_towerHad;
        std::vector< std::vector<float> > _cl5x7_towerEt;
        std::vector< std::vector<float> > _cl5x7_towerEgEt;
        std::vector< std::vector<int> >   _cl5x7_towerIeta;
        std::vector< std::vector<int> >   _cl5x7_towerIphi;
        std::vector< std::vector<int> >   _cl5x7_towerIem;
        std::vector< std::vector<int> >   _cl5x7_towerIhad;
        std::vector< std::vector<int> >   _cl5x7_towerIet;
        std::vector< std::vector<int> >   _cl5x7_towerEgIet;
        std::vector< std::vector<int> >   _cl5x7_towerNeg;

        std::vector<bool>  _cl3x7_barrelSeeded;
        std::vector<int>   _cl3x7_nHits;
        std::vector<int>   _cl3x7_nEGs;
        std::vector<int>   _cl3x7_seedIeta;
        std::vector<int>   _cl3x7_seedIphi;
        std::vector<float> _cl3x7_seedEta;
        std::vector<float> _cl3x7_seedPhi;
        std::vector<bool>  _cl3x7_isBarrel;
        std::vector<bool>  _cl3x7_isOverlap;
        std::vector<bool>  _cl3x7_isEndcap;
        std::vector<bool>  _cl3x7_isPhiFlipped;
        std::vector<int>   _cl3x7_tauMatchIdx;
        std::vector<int>   _cl3x7_jetMatchIdx;
        std::vector<int>   _cl3x7_cl3dMatchIdx;
        std::vector<float> _cl3x7_totalEm;
        std::vector<float> _cl3x7_totalHad;
        std::vector<float> _cl3x7_totalEt;
        std::vector<float> _cl3x7_totalEgEt;
        std::vector<int> _cl3x7_totalIem;
        std::vector<int> _cl3x7_totalIhad;
        std::vector<int> _cl3x7_totalIet;
        std::vector<int> _cl3x7_totalEgIet;
        std::vector< std::vector<float> > _cl3x7_towerEta;
        std::vector< std::vector<float> > _cl3x7_towerPhi;
        std::vector< std::vector<float> > _cl3x7_towerEm;
        std::vector< std::vector<float> > _cl3x7_towerHad;
        std::vector< std::vector<float> > _cl3x7_towerEt;
        std::vector< std::vector<float> > _cl3x7_towerEgEt;
        std::vector< std::vector<int> >   _cl3x7_towerIeta;
        std::vector< std::vector<int> >   _cl3x7_towerIphi;
        std::vector< std::vector<int> >   _cl3x7_towerIem;
        std::vector< std::vector<int> >   _cl3x7_towerIhad;
        std::vector< std::vector<int> >   _cl3x7_towerIet;
        std::vector< std::vector<int> >   _cl3x7_towerEgIet;
        std::vector< std::vector<int> >   _cl3x7_towerNeg;

        std::vector<bool>  _cl3x5_barrelSeeded;
        std::vector<int>   _cl3x5_nHits;
        std::vector<int>   _cl3x5_nEGs;
        std::vector<int>   _cl3x5_seedIeta;
        std::vector<int>   _cl3x5_seedIphi;
        std::vector<float> _cl3x5_seedEta;
        std::vector<float> _cl3x5_seedPhi;
        std::vector<bool>  _cl3x5_isBarrel;
        std::vector<bool>  _cl3x5_isOverlap;
        std::vector<bool>  _cl3x5_isEndcap;
        std::vector<bool>  _cl3x5_isPhiFlipped;
        std::vector<int>   _cl3x5_tauMatchIdx;
        std::vector<int>   _cl3x5_jetMatchIdx;
        std::vector<int>   _cl3x5_cl3dMatchIdx;
        std::vector<float> _cl3x5_totalEm;
        std::vector<float> _cl3x5_totalHad;
        std::vector<float> _cl3x5_totalEt;
        std::vector<float> _cl3x5_totalEgEt;
        std::vector<int>   _cl3x5_totalIem;
        std::vector<int>   _cl3x5_totalIhad;
        std::vector<int>   _cl3x5_totalIet;
        std::vector<int>   _cl3x5_totalEgIet;
        std::vector< std::vector<float> > _cl3x5_towerEta;
        std::vector< std::vector<float> > _cl3x5_towerPhi;
        std::vector< std::vector<float> > _cl3x5_towerEm;
        std::vector< std::vector<float> > _cl3x5_towerHad;
        std::vector< std::vector<float> > _cl3x5_towerEt;
        std::vector< std::vector<float> > _cl3x5_towerEgEt;
        std::vector< std::vector<int> >   _cl3x5_towerIeta;
        std::vector< std::vector<int> >   _cl3x5_towerIphi;
        std::vector< std::vector<int> >   _cl3x5_towerIem;
        std::vector< std::vector<int> >   _cl3x5_towerIhad;
        std::vector< std::vector<int> >   _cl3x5_towerIet;
        std::vector< std::vector<int> >   _cl3x5_towerEgIet;
        std::vector< std::vector<int> >   _cl3x5_towerNeg;
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
      CaloClusters5x7Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters5x7"))),
      CaloClusters3x7Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters3x7"))),
      CaloClusters3x5Token(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClusters3x5"))),
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
    _tau_Idx.clear();

    _jet_eta.clear();
    _jet_phi.clear();
    _jet_pt.clear();
    _jet_e.clear();
    _jet_eEm.clear();
    _jet_eHad.clear();
    _jet_eInv.clear();
    _jet_Idx.clear();

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
    _cl3d_puid.clear();
    _cl3d_puidscore.clear();
    _cl3d_pionid.clear();
    _cl3d_pionidscore.clear();
    _cl3d_tauMatchIdx.clear();
    _cl3d_jetMatchIdx.clear();

    _cl9x9_barrelSeeded.clear();
    _cl9x9_nHits.clear();
    _cl9x9_nEGs.clear();
    _cl9x9_seedIeta.clear();
    _cl9x9_seedIphi.clear();
    _cl9x9_seedEta.clear();
    _cl9x9_seedPhi.clear();
    _cl9x9_isBarrel.clear();
    _cl9x9_isOverlap.clear();
    _cl9x9_isEndcap.clear();
    _cl9x9_isPhiFlipped.clear();
    _cl9x9_tauMatchIdx.clear();
    _cl9x9_jetMatchIdx.clear();
    _cl9x9_cl3dMatchIdx.clear();
    _cl9x9_totalEm.clear();
    _cl9x9_totalHad.clear();
    _cl9x9_totalEt.clear();
    _cl9x9_totalEgEt.clear();
    _cl9x9_totalIem.clear();
    _cl9x9_totalIhad.clear();
    _cl9x9_totalIet.clear();
    _cl9x9_totalEgIet.clear();
    _cl9x9_towerEta.clear();
    _cl9x9_towerPhi.clear();
    _cl9x9_towerEm.clear();
    _cl9x9_towerHad.clear();
    _cl9x9_towerEt.clear();
    _cl9x9_towerEgEt.clear();
    _cl9x9_towerIeta.clear();
    _cl9x9_towerIphi.clear();
    _cl9x9_towerIem.clear();
    _cl9x9_towerIhad.clear();
    _cl9x9_towerIet.clear();
    _cl9x9_towerEgIet.clear();
    _cl9x9_towerNeg.clear();

    _cl7x7_barrelSeeded.clear();
    _cl7x7_nHits.clear();
    _cl7x7_nEGs.clear();
    _cl7x7_seedIeta.clear();
    _cl7x7_seedIphi.clear();
    _cl7x7_seedEta.clear();
    _cl7x7_seedPhi.clear();
    _cl7x7_isBarrel.clear();
    _cl7x7_isOverlap.clear();
    _cl7x7_isEndcap.clear();
    _cl7x7_isPhiFlipped.clear();
    _cl7x7_tauMatchIdx.clear();
    _cl7x7_jetMatchIdx.clear();
    _cl7x7_cl3dMatchIdx.clear();
    _cl7x7_totalEm.clear();
    _cl7x7_totalHad.clear();
    _cl7x7_totalEt.clear();
    _cl7x7_totalEgEt.clear();
    _cl7x7_totalIem.clear();
    _cl7x7_totalIhad.clear();
    _cl7x7_totalIet.clear();
    _cl7x7_totalEgIet.clear();
    _cl7x7_towerEta.clear();
    _cl7x7_towerPhi.clear();
    _cl7x7_towerEm.clear();
    _cl7x7_towerHad.clear();
    _cl7x7_towerEt.clear();
    _cl7x7_towerEgEt.clear();
    _cl7x7_towerIeta.clear();
    _cl7x7_towerIphi.clear();
    _cl7x7_towerIem.clear();
    _cl7x7_towerIhad.clear();
    _cl7x7_towerIet.clear();
    _cl7x7_towerEgIet.clear();
    _cl7x7_towerNeg.clear();

    _cl5x5_barrelSeeded.clear();
    _cl5x5_nHits.clear();
    _cl5x5_nEGs.clear();
    _cl5x5_seedIeta.clear();
    _cl5x5_seedIphi.clear();
    _cl5x5_seedEta.clear();
    _cl5x5_seedPhi.clear();
    _cl5x5_isBarrel.clear();
    _cl5x5_isOverlap.clear();
    _cl5x5_isEndcap.clear();
    _cl5x5_isPhiFlipped.clear();
    _cl5x5_tauMatchIdx.clear();
    _cl5x5_jetMatchIdx.clear();
    _cl5x5_cl3dMatchIdx.clear();
    _cl5x5_totalEm.clear();
    _cl5x5_totalHad.clear();
    _cl5x5_totalEt.clear();
    _cl5x5_totalEgEt.clear();
    _cl5x5_totalIem.clear();
    _cl5x5_totalIhad.clear();
    _cl5x5_totalIet.clear();
    _cl5x5_totalEgIet.clear();
    _cl5x5_towerEta.clear();
    _cl5x5_towerPhi.clear();
    _cl5x5_towerEm.clear();
    _cl5x5_towerHad.clear();
    _cl5x5_towerEt.clear();
    _cl5x5_towerEgEt.clear();
    _cl5x5_towerIeta.clear();
    _cl5x5_towerIphi.clear();
    _cl5x5_towerIem.clear();
    _cl5x5_towerIhad.clear();
    _cl5x5_towerIet.clear();
    _cl5x5_towerEgIet.clear();
    _cl5x5_towerNeg.clear();

    _cl5x9_barrelSeeded.clear();
    _cl5x9_nHits.clear();
    _cl5x9_nEGs.clear();
    _cl5x9_seedIeta.clear();
    _cl5x9_seedIphi.clear();
    _cl5x9_seedEta.clear();
    _cl5x9_seedPhi.clear();
    _cl5x9_isBarrel.clear();
    _cl5x9_isOverlap.clear();
    _cl5x9_isEndcap.clear();
    _cl5x9_isPhiFlipped.clear();
    _cl5x9_tauMatchIdx.clear();
    _cl5x9_jetMatchIdx.clear();
    _cl5x9_cl3dMatchIdx.clear();
    _cl5x9_totalEm.clear();
    _cl5x9_totalHad.clear();
    _cl5x9_totalEt.clear();
    _cl5x9_totalEgEt.clear();
    _cl5x9_totalIem.clear();
    _cl5x9_totalIhad.clear();
    _cl5x9_totalIet.clear();
    _cl5x9_totalEgIet.clear();
    _cl5x9_towerEta.clear();
    _cl5x9_towerPhi.clear();
    _cl5x9_towerEm.clear();
    _cl5x9_towerHad.clear();
    _cl5x9_towerEt.clear();
    _cl5x9_towerEgEt.clear();
    _cl5x9_towerIeta.clear();
    _cl5x9_towerIphi.clear();
    _cl5x9_towerIem.clear();
    _cl5x9_towerIhad.clear();
    _cl5x9_towerIet.clear();
    _cl5x9_towerEgIet.clear();
    _cl5x9_towerNeg.clear();

    _cl5x7_barrelSeeded.clear();
    _cl5x7_nHits.clear();
    _cl5x7_nEGs.clear();
    _cl5x7_seedIeta.clear();
    _cl5x7_seedIphi.clear();
    _cl5x7_seedEta.clear();
    _cl5x7_seedPhi.clear();
    _cl5x7_isBarrel.clear();
    _cl5x7_isOverlap.clear();
    _cl5x7_isEndcap.clear();
    _cl5x7_isPhiFlipped.clear();
    _cl5x7_tauMatchIdx.clear();
    _cl5x7_jetMatchIdx.clear();
    _cl5x7_cl3dMatchIdx.clear();
    _cl5x7_totalEm.clear();
    _cl5x7_totalHad.clear();
    _cl5x7_totalEt.clear();
    _cl5x7_totalEgEt.clear();
    _cl5x7_totalIem.clear();
    _cl5x7_totalIhad.clear();
    _cl5x7_totalIet.clear();
    _cl5x7_totalEgIet.clear();
    _cl5x7_towerEta.clear();
    _cl5x7_towerPhi.clear();
    _cl5x7_towerEm.clear();
    _cl5x7_towerHad.clear();
    _cl5x7_towerEt.clear();
    _cl5x7_towerEgEt.clear();
    _cl5x7_towerIeta.clear();
    _cl5x7_towerIphi.clear();
    _cl5x7_towerIem.clear();
    _cl5x7_towerIhad.clear();
    _cl5x7_towerIet.clear();
    _cl5x7_towerEgIet.clear();
    _cl5x7_towerNeg.clear();

    _cl3x7_barrelSeeded.clear();
    _cl3x7_nHits.clear();
    _cl3x7_nEGs.clear();
    _cl3x7_seedIeta.clear();
    _cl3x7_seedIphi.clear();
    _cl3x7_seedEta.clear();
    _cl3x7_seedPhi.clear();
    _cl3x7_isBarrel.clear();
    _cl3x7_isOverlap.clear();
    _cl3x7_isEndcap.clear();
    _cl3x7_isPhiFlipped.clear();
    _cl3x7_tauMatchIdx.clear();
    _cl3x7_jetMatchIdx.clear();
    _cl3x7_cl3dMatchIdx.clear();
    _cl3x7_totalEm.clear();
    _cl3x7_totalHad.clear();
    _cl3x7_totalEt.clear();
    _cl3x7_totalEgEt.clear();
    _cl3x7_totalIem.clear();
    _cl3x7_totalIhad.clear();
    _cl3x7_totalIet.clear();
    _cl3x7_totalEgIet.clear();
    _cl3x7_towerEta.clear();
    _cl3x7_towerPhi.clear();
    _cl3x7_towerEm.clear();
    _cl3x7_towerHad.clear();
    _cl3x7_towerEt.clear();
    _cl3x7_towerEgEt.clear();
    _cl3x7_towerIeta.clear();
    _cl3x7_towerIphi.clear();
    _cl3x7_towerIem.clear();
    _cl3x7_towerIhad.clear();
    _cl3x7_towerIet.clear();
    _cl3x7_towerEgIet.clear();
    _cl3x7_towerNeg.clear();

    _cl3x5_barrelSeeded.clear();
    _cl3x5_nHits.clear();
    _cl3x5_nEGs.clear();
    _cl3x5_seedIeta.clear();
    _cl3x5_seedIphi.clear();
    _cl3x5_seedEta.clear();
    _cl3x5_seedPhi.clear();
    _cl3x5_isBarrel.clear();
    _cl3x5_isOverlap.clear();
    _cl3x5_isEndcap.clear();
    _cl3x5_isPhiFlipped.clear();
    _cl3x5_tauMatchIdx.clear();
    _cl3x5_jetMatchIdx.clear();
    _cl3x5_cl3dMatchIdx.clear();
    _cl3x5_totalEm.clear();
    _cl3x5_totalHad.clear();
    _cl3x5_totalEt.clear();
    _cl3x5_totalEgEt.clear();
    _cl3x5_totalIem.clear();
    _cl3x5_totalIhad.clear();
    _cl3x5_totalIet.clear();
    _cl3x5_totalEgIet.clear();
    _cl3x5_towerEta.clear();
    _cl3x5_towerPhi.clear();
    _cl3x5_towerEm.clear();
    _cl3x5_towerHad.clear();
    _cl3x5_towerEt.clear();
    _cl3x5_towerEgEt.clear();
    _cl3x5_towerIeta.clear();
    _cl3x5_towerIphi.clear();
    _cl3x5_towerIem.clear();
    _cl3x5_towerIhad.clear();
    _cl3x5_towerIet.clear();
    _cl3x5_towerEgIet.clear();
    _cl3x5_towerNeg.clear();
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
    _tree -> Branch("tau_Idx",      &_tau_Idx);

    _tree -> Branch("jet_eta",  &_jet_eta);
    _tree -> Branch("jet_phi",  &_jet_phi);
    _tree -> Branch("jet_pt",   &_jet_pt);
    _tree -> Branch("jet_e",    &_jet_e);
    _tree -> Branch("jet_eEm",  &_jet_eEm);
    _tree -> Branch("jet_eHad", &_jet_eHad);
    _tree -> Branch("jet_eInv", &_jet_eInv);
    _tree -> Branch("jet_Idx",  &_jet_Idx);

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
    _tree -> Branch("cl3d_puid",             &_cl3d_puid);
    _tree -> Branch("cl3d_puidscore",        &_cl3d_puidscore);
    _tree -> Branch("cl3d_pionid",           &_cl3d_pionid);
    _tree -> Branch("cl3d_pionidscore",      &_cl3d_pionidscore);
    _tree -> Branch("cl3d_tauMatchIdx",      &_cl3d_tauMatchIdx);
    _tree -> Branch("cl3d_jetMatchIdx",      &_cl3d_jetMatchIdx);

    _tree -> Branch("cl9x9_barrelSeeded", &_cl9x9_barrelSeeded);
    _tree -> Branch("cl9x9_nHits",        &_cl9x9_nHits);
    _tree -> Branch("cl9x9_nEGs",         &_cl9x9_nEGs);
    _tree -> Branch("cl9x9_seedIeta",     &_cl9x9_seedIeta);
    _tree -> Branch("cl9x9_seedIphi",     &_cl9x9_seedIphi);
    _tree -> Branch("cl9x9_seedEta",      &_cl9x9_seedEta);
    _tree -> Branch("cl9x9_seedPhi",      &_cl9x9_seedPhi);
    _tree -> Branch("cl9x9_isBarrel",     &_cl9x9_isBarrel);
    _tree -> Branch("cl9x9_isOverlap",    &_cl9x9_isOverlap);
    _tree -> Branch("cl9x9_isEndcap",     &_cl9x9_isEndcap);
    _tree -> Branch("cl9x9_isPhiFlipped", &_cl9x9_isPhiFlipped);
    _tree -> Branch("cl9x9_tauMatchIdx",  &_cl9x9_tauMatchIdx);
    _tree -> Branch("cl9x9_jetMatchIdx",  &_cl9x9_jetMatchIdx);
    _tree -> Branch("cl9x9_cl3dMatchIdx", &_cl9x9_cl3dMatchIdx);
    _tree -> Branch("cl9x9_totalEm",      &_cl9x9_totalEm);
    _tree -> Branch("cl9x9_totalHad",     &_cl9x9_totalHad);
    _tree -> Branch("cl9x9_totalEt",      &_cl9x9_totalEt);
    _tree -> Branch("cl9x9_totalEgEt",    &_cl9x9_totalEgEt);
    _tree -> Branch("cl9x9_totalIem",     &_cl9x9_totalIem);
    _tree -> Branch("cl9x9_totalIhad",    &_cl9x9_totalIhad);
    _tree -> Branch("cl9x9_totalIet",     &_cl9x9_totalIet);
    _tree -> Branch("cl9x9_totalEgIet",   &_cl9x9_totalEgIet);
    _tree -> Branch("cl9x9_towerEta",     &_cl9x9_towerEta);
    _tree -> Branch("cl9x9_towerPhi",     &_cl9x9_towerPhi);
    _tree -> Branch("cl9x9_towerEm",      &_cl9x9_towerEm);
    _tree -> Branch("cl9x9_towerHad",     &_cl9x9_towerHad);
    _tree -> Branch("cl9x9_towerEt",      &_cl9x9_towerEt);
    _tree -> Branch("cl9x9_towerEgEt",    &_cl9x9_towerEgEt);
    _tree -> Branch("cl9x9_towerIeta",    &_cl9x9_towerIeta);
    _tree -> Branch("cl9x9_towerIphi",    &_cl9x9_towerIphi);
    _tree -> Branch("cl9x9_towerIem",     &_cl9x9_towerIem);
    _tree -> Branch("cl9x9_towerIhad",    &_cl9x9_towerIhad);
    _tree -> Branch("cl9x9_towerIet",     &_cl9x9_towerIet);
    _tree -> Branch("cl9x9_towerEgIet",   &_cl9x9_towerEgIet);
    _tree -> Branch("cl9x9_towerNeg",     &_cl9x9_towerNeg);

    _tree -> Branch("cl7x7_barrelSeeded", &_cl7x7_barrelSeeded);
    _tree -> Branch("cl7x7_nHits",        &_cl7x7_nHits);
    _tree -> Branch("cl7x7_nEGs",         &_cl7x7_nEGs);
    _tree -> Branch("cl7x7_seedIeta",     &_cl7x7_seedIeta);
    _tree -> Branch("cl7x7_seedIphi",     &_cl7x7_seedIphi);
    _tree -> Branch("cl7x7_seedEta",      &_cl7x7_seedEta);
    _tree -> Branch("cl7x7_seedPhi",      &_cl7x7_seedPhi);
    _tree -> Branch("cl7x7_isBarrel",     &_cl7x7_isBarrel);
    _tree -> Branch("cl7x7_isOverlap",    &_cl7x7_isOverlap);
    _tree -> Branch("cl7x7_isEndcap",     &_cl7x7_isEndcap);
    _tree -> Branch("cl7x7_isPhiFlipped", &_cl7x7_isPhiFlipped);
    _tree -> Branch("cl7x7_tauMatchIdx",  &_cl7x7_tauMatchIdx);
    _tree -> Branch("cl7x7_jetMatchIdx",  &_cl7x7_jetMatchIdx);
    _tree -> Branch("cl7x7_cl3dMatchIdx", &_cl7x7_cl3dMatchIdx);
    _tree -> Branch("cl7x7_totalEm",      &_cl7x7_totalEm);
    _tree -> Branch("cl7x7_totalHad",     &_cl7x7_totalHad);
    _tree -> Branch("cl7x7_totalEt",      &_cl7x7_totalEt);
    _tree -> Branch("cl7x7_totalEgEt",    &_cl7x7_totalEgEt);
    _tree -> Branch("cl7x7_totalIem",     &_cl7x7_totalIem);
    _tree -> Branch("cl7x7_totalIhad",    &_cl7x7_totalIhad);
    _tree -> Branch("cl7x7_totalIet",     &_cl7x7_totalIet);
    _tree -> Branch("cl7x7_totalEgIet",   &_cl7x7_totalEgIet);
    _tree -> Branch("cl7x7_towerEta",     &_cl7x7_towerEta);
    _tree -> Branch("cl7x7_towerPhi",     &_cl7x7_towerPhi);
    _tree -> Branch("cl7x7_towerEm",      &_cl7x7_towerEm);
    _tree -> Branch("cl7x7_towerHad",     &_cl7x7_towerHad);
    _tree -> Branch("cl7x7_towerEt",      &_cl7x7_towerEt);
    _tree -> Branch("cl7x7_towerEgEt",    &_cl7x7_towerEgEt);
    _tree -> Branch("cl7x7_towerIeta",    &_cl7x7_towerIeta);
    _tree -> Branch("cl7x7_towerIphi",    &_cl7x7_towerIphi);
    _tree -> Branch("cl7x7_towerIem",     &_cl7x7_towerIem);
    _tree -> Branch("cl7x7_towerIhad",    &_cl7x7_towerIhad);
    _tree -> Branch("cl7x7_towerIet",     &_cl7x7_towerIet);
    _tree -> Branch("cl7x7_towerEgIet",   &_cl7x7_towerEgIet);
    _tree -> Branch("cl7x7_towerNeg",     &_cl7x7_towerNeg);

    _tree -> Branch("cl5x5_barrelSeeded", &_cl5x5_barrelSeeded);
    _tree -> Branch("cl5x5_nHits",        &_cl5x5_nHits);
    _tree -> Branch("cl5x5_nEGs",         &_cl5x5_nEGs);
    _tree -> Branch("cl5x5_seedIeta",     &_cl5x5_seedIeta);
    _tree -> Branch("cl5x5_seedIphi",     &_cl5x5_seedIphi);
    _tree -> Branch("cl5x5_seedEta",      &_cl5x5_seedEta);
    _tree -> Branch("cl5x5_seedPhi",      &_cl5x5_seedPhi);
    _tree -> Branch("cl5x5_isBarrel",     &_cl5x5_isBarrel);
    _tree -> Branch("cl5x5_isOverlap",    &_cl5x5_isOverlap);
    _tree -> Branch("cl5x5_isEndcap",     &_cl5x5_isEndcap);
    _tree -> Branch("cl5x5_isPhiFlipped", &_cl5x5_isPhiFlipped);
    _tree -> Branch("cl5x5_tauMatchIdx",  &_cl5x5_tauMatchIdx);
    _tree -> Branch("cl5x5_jetMatchIdx",  &_cl5x5_jetMatchIdx);
    _tree -> Branch("cl5x5_cl3dMatchIdx", &_cl5x5_cl3dMatchIdx);
    _tree -> Branch("cl5x5_totalEm",      &_cl5x5_totalEm);
    _tree -> Branch("cl5x5_totalHad",     &_cl5x5_totalHad);
    _tree -> Branch("cl5x5_totalEt",      &_cl5x5_totalEt);
    _tree -> Branch("cl5x5_totalEgEt",    &_cl5x5_totalEgEt);
    _tree -> Branch("cl5x5_totalIem",     &_cl5x5_totalIem);
    _tree -> Branch("cl5x5_totalIhad",    &_cl5x5_totalIhad);
    _tree -> Branch("cl5x5_totalIet",     &_cl5x5_totalIet);
    _tree -> Branch("cl5x5_totalEgIet",   &_cl5x5_totalEgIet);
    _tree -> Branch("cl5x5_towerEta",     &_cl5x5_towerEta);
    _tree -> Branch("cl5x5_towerPhi",     &_cl5x5_towerPhi);
    _tree -> Branch("cl5x5_towerEm",      &_cl5x5_towerEm);
    _tree -> Branch("cl5x5_towerHad",     &_cl5x5_towerHad);
    _tree -> Branch("cl5x5_towerEt",      &_cl5x5_towerEt);
    _tree -> Branch("cl5x5_towerEgEt",    &_cl5x5_towerEgEt);
    _tree -> Branch("cl5x5_towerIeta",    &_cl5x5_towerIeta);
    _tree -> Branch("cl5x5_towerIphi",    &_cl5x5_towerIphi);
    _tree -> Branch("cl5x5_towerIem",     &_cl5x5_towerIem);
    _tree -> Branch("cl5x5_towerIhad",    &_cl5x5_towerIhad);
    _tree -> Branch("cl5x5_towerIet",     &_cl5x5_towerIet);
    _tree -> Branch("cl5x5_towerEgIet",   &_cl5x5_towerEgIet);
    _tree -> Branch("cl5x5_towerNeg",     &_cl5x5_towerNeg);

    _tree -> Branch("cl5x9_barrelSeeded", &_cl5x9_barrelSeeded);
    _tree -> Branch("cl5x9_nHits",        &_cl5x9_nHits);
    _tree -> Branch("cl5x9_nEGs",         &_cl5x9_nEGs);
    _tree -> Branch("cl5x9_seedIeta",     &_cl5x9_seedIeta);
    _tree -> Branch("cl5x9_seedIphi",     &_cl5x9_seedIphi);
    _tree -> Branch("cl5x9_seedEta",      &_cl5x9_seedEta);
    _tree -> Branch("cl5x9_seedPhi",      &_cl5x9_seedPhi);
    _tree -> Branch("cl5x9_isBarrel",     &_cl5x9_isBarrel);
    _tree -> Branch("cl5x9_isOverlap",    &_cl5x9_isOverlap);
    _tree -> Branch("cl5x9_isEndcap",     &_cl5x9_isEndcap);
    _tree -> Branch("cl5x9_isPhiFlipped", &_cl5x9_isPhiFlipped);
    _tree -> Branch("cl5x9_tauMatchIdx",  &_cl5x9_tauMatchIdx);
    _tree -> Branch("cl5x9_jetMatchIdx",  &_cl5x9_jetMatchIdx);
    _tree -> Branch("cl5x9_cl3dMatchIdx", &_cl5x9_cl3dMatchIdx);
    _tree -> Branch("cl5x9_totalEm",      &_cl5x9_totalEm);
    _tree -> Branch("cl5x9_totalHad",     &_cl5x9_totalHad);
    _tree -> Branch("cl5x9_totalEt",      &_cl5x9_totalEt);
    _tree -> Branch("cl5x9_totalEgEt",    &_cl5x9_totalEgEt);
    _tree -> Branch("cl5x9_totalIem",     &_cl5x9_totalIem);
    _tree -> Branch("cl5x9_totalIhad",    &_cl5x9_totalIhad);
    _tree -> Branch("cl5x9_totalIet",     &_cl5x9_totalIet);
    _tree -> Branch("cl5x9_totalEgIet",   &_cl5x9_totalEgIet);
    _tree -> Branch("cl5x9_towerEta",     &_cl5x9_towerEta);
    _tree -> Branch("cl5x9_towerPhi",     &_cl5x9_towerPhi);
    _tree -> Branch("cl5x9_towerEm",      &_cl5x9_towerEm);
    _tree -> Branch("cl5x9_towerHad",     &_cl5x9_towerHad);
    _tree -> Branch("cl5x9_towerEt",      &_cl5x9_towerEt);
    _tree -> Branch("cl5x9_towerEgEt",    &_cl5x9_towerEgEt);
    _tree -> Branch("cl5x9_towerIeta",    &_cl5x9_towerIeta);
    _tree -> Branch("cl5x9_towerIphi",    &_cl5x9_towerIphi);
    _tree -> Branch("cl5x9_towerIem",     &_cl5x9_towerIem);
    _tree -> Branch("cl5x9_towerIhad",    &_cl5x9_towerIhad);
    _tree -> Branch("cl5x9_towerIet",     &_cl5x9_towerIet);
    _tree -> Branch("cl5x9_towerEgIet",   &_cl5x9_towerEgIet);
    _tree -> Branch("cl5x9_towerNeg",     &_cl5x9_towerNeg);

    _tree -> Branch("cl5x7_barrelSeeded", &_cl5x7_barrelSeeded);
    _tree -> Branch("cl5x7_nHits",        &_cl5x7_nHits);
    _tree -> Branch("cl5x7_nEGs",         &_cl5x7_nEGs);
    _tree -> Branch("cl5x7_seedIeta",     &_cl5x7_seedIeta);
    _tree -> Branch("cl5x7_seedIphi",     &_cl5x7_seedIphi);
    _tree -> Branch("cl5x7_seedEta",      &_cl5x7_seedEta);
    _tree -> Branch("cl5x7_seedPhi",      &_cl5x7_seedPhi);
    _tree -> Branch("cl5x7_isBarrel",     &_cl5x7_isBarrel);
    _tree -> Branch("cl5x7_isOverlap",    &_cl5x7_isOverlap);
    _tree -> Branch("cl5x7_isEndcap",     &_cl5x7_isEndcap);
    _tree -> Branch("cl5x7_isPhiFlipped", &_cl5x7_isPhiFlipped);
    _tree -> Branch("cl5x7_tauMatchIdx",  &_cl5x7_tauMatchIdx);
    _tree -> Branch("cl5x7_jetMatchIdx",  &_cl5x7_jetMatchIdx);
    _tree -> Branch("cl5x7_cl3dMatchIdx", &_cl5x7_cl3dMatchIdx);
    _tree -> Branch("cl5x7_totalEm",      &_cl5x7_totalEm);
    _tree -> Branch("cl5x7_totalHad",     &_cl5x7_totalHad);
    _tree -> Branch("cl5x7_totalEt",      &_cl5x7_totalEt);
    _tree -> Branch("cl5x7_totalEgEt",    &_cl5x7_totalEgEt);
    _tree -> Branch("cl5x7_totalIem",     &_cl5x7_totalIem);
    _tree -> Branch("cl5x7_totalIhad",    &_cl5x7_totalIhad);
    _tree -> Branch("cl5x7_totalIet",     &_cl5x7_totalIet);
    _tree -> Branch("cl5x7_totalEgIet",   &_cl5x7_totalEgIet);
    _tree -> Branch("cl5x7_towerEta",     &_cl5x7_towerEta);
    _tree -> Branch("cl5x7_towerPhi",     &_cl5x7_towerPhi);
    _tree -> Branch("cl5x7_towerEm",      &_cl5x7_towerEm);
    _tree -> Branch("cl5x7_towerHad",     &_cl5x7_towerHad);
    _tree -> Branch("cl5x7_towerEt",      &_cl5x7_towerEt);
    _tree -> Branch("cl5x7_towerEgEt",    &_cl5x7_towerEgEt);
    _tree -> Branch("cl5x7_towerIeta",    &_cl5x7_towerIeta);
    _tree -> Branch("cl5x7_towerIphi",    &_cl5x7_towerIphi);
    _tree -> Branch("cl5x7_towerIem",     &_cl5x7_towerIem);
    _tree -> Branch("cl5x7_towerIhad",    &_cl5x7_towerIhad);
    _tree -> Branch("cl5x7_towerIet",     &_cl5x7_towerIet);
    _tree -> Branch("cl5x7_towerEgIet",   &_cl5x7_towerEgIet);
    _tree -> Branch("cl5x7_towerNeg",     &_cl5x7_towerNeg);

    _tree -> Branch("cl3x7_barrelSeeded", &_cl3x7_barrelSeeded);
    _tree -> Branch("cl3x7_nHits",        &_cl3x7_nHits);
    _tree -> Branch("cl3x7_nEGs",         &_cl3x7_nEGs);
    _tree -> Branch("cl3x7_seedIeta",     &_cl3x7_seedIeta);
    _tree -> Branch("cl3x7_seedIphi",     &_cl3x7_seedIphi);
    _tree -> Branch("cl3x7_seedEta",      &_cl3x7_seedEta);
    _tree -> Branch("cl3x7_seedPhi",      &_cl3x7_seedPhi);
    _tree -> Branch("cl3x7_isBarrel",     &_cl3x7_isBarrel);
    _tree -> Branch("cl3x7_isOverlap",    &_cl3x7_isOverlap);
    _tree -> Branch("cl3x7_isEndcap",     &_cl3x7_isEndcap);
    _tree -> Branch("cl3x7_isPhiFlipped", &_cl3x7_isPhiFlipped);
    _tree -> Branch("cl3x7_tauMatchIdx",  &_cl3x7_tauMatchIdx);
    _tree -> Branch("cl3x7_jetMatchIdx",  &_cl3x7_jetMatchIdx);
    _tree -> Branch("cl3x7_cl3dMatchIdx", &_cl3x7_cl3dMatchIdx);
    _tree -> Branch("cl3x7_totalEm",      &_cl3x7_totalEm);
    _tree -> Branch("cl3x7_totalHad",     &_cl3x7_totalHad);
    _tree -> Branch("cl3x7_totalEt",      &_cl3x7_totalEt);
    _tree -> Branch("cl3x7_totalEgEt",    &_cl3x7_totalEgEt);
    _tree -> Branch("cl3x7_totalIem",     &_cl3x7_totalIem);
    _tree -> Branch("cl3x7_totalIhad",    &_cl3x7_totalIhad);
    _tree -> Branch("cl3x7_totalIet",     &_cl3x7_totalIet);
    _tree -> Branch("cl3x7_totalEgIet",   &_cl3x7_totalEgIet);
    _tree -> Branch("cl3x7_towerEta",     &_cl3x7_towerEta);
    _tree -> Branch("cl3x7_towerPhi",     &_cl3x7_towerPhi);
    _tree -> Branch("cl3x7_towerEm",      &_cl3x7_towerEm);
    _tree -> Branch("cl3x7_towerHad",     &_cl3x7_towerHad);
    _tree -> Branch("cl3x7_towerEt",      &_cl3x7_towerEt);
    _tree -> Branch("cl3x7_towerEgEt",    &_cl3x7_towerEgEt);
    _tree -> Branch("cl3x7_towerIeta",    &_cl3x7_towerIeta);
    _tree -> Branch("cl3x7_towerIphi",    &_cl3x7_towerIphi);
    _tree -> Branch("cl3x7_towerIem",     &_cl3x7_towerIem);
    _tree -> Branch("cl3x7_towerIhad",    &_cl3x7_towerIhad);
    _tree -> Branch("cl3x7_towerIet",     &_cl3x7_towerIet);
    _tree -> Branch("cl3x7_towerEgIet",   &_cl3x7_towerEgIet);
    _tree -> Branch("cl3x7_towerNeg",     &_cl3x7_towerNeg);

    _tree -> Branch("cl3x5_barrelSeeded", &_cl3x5_barrelSeeded);
    _tree -> Branch("cl3x5_nHits",        &_cl3x5_nHits);
    _tree -> Branch("cl3x5_nEGs",         &_cl3x5_nEGs);
    _tree -> Branch("cl3x5_seedIeta",     &_cl3x5_seedIeta);
    _tree -> Branch("cl3x5_seedIphi",     &_cl3x5_seedIphi);
    _tree -> Branch("cl3x5_seedEta",      &_cl3x5_seedEta);
    _tree -> Branch("cl3x5_seedPhi",      &_cl3x5_seedPhi);
    _tree -> Branch("cl3x5_isBarrel",     &_cl3x5_isBarrel);
    _tree -> Branch("cl3x5_isOverlap",    &_cl3x5_isOverlap);
    _tree -> Branch("cl3x5_isEndcap",     &_cl3x5_isEndcap);
    _tree -> Branch("cl3x5_isPhiFlipped", &_cl3x5_isPhiFlipped);
    _tree -> Branch("cl3x5_tauMatchIdx",  &_cl3x5_tauMatchIdx);
    _tree -> Branch("cl3x5_jetMatchIdx",  &_cl3x5_jetMatchIdx);
    _tree -> Branch("cl3x5_cl3dMatchIdx", &_cl3x5_cl3dMatchIdx);
    _tree -> Branch("cl3x5_totalEm",      &_cl3x5_totalEm);
    _tree -> Branch("cl3x5_totalHad",     &_cl3x5_totalHad);
    _tree -> Branch("cl3x5_totalEt",      &_cl3x5_totalEt);
    _tree -> Branch("cl3x5_totalEgEt",    &_cl3x5_totalEgEt);
    _tree -> Branch("cl3x5_totalIem",     &_cl3x5_totalIem);
    _tree -> Branch("cl3x5_totalIhad",    &_cl3x5_totalIhad);
    _tree -> Branch("cl3x5_totalIet",     &_cl3x5_totalIet);
    _tree -> Branch("cl3x5_totalEgIet",   &_cl3x5_totalEgIet);
    _tree -> Branch("cl3x5_towerEta",     &_cl3x5_towerEta);
    _tree -> Branch("cl3x5_towerPhi",     &_cl3x5_towerPhi);
    _tree -> Branch("cl3x5_towerEm",      &_cl3x5_towerEm);
    _tree -> Branch("cl3x5_towerHad",     &_cl3x5_towerHad);
    _tree -> Branch("cl3x5_towerEt",      &_cl3x5_towerEt);
    _tree -> Branch("cl3x5_towerEgEt",    &_cl3x5_towerEgEt);
    _tree -> Branch("cl3x5_towerIeta",    &_cl3x5_towerIeta);
    _tree -> Branch("cl3x5_towerIphi",    &_cl3x5_towerIphi);
    _tree -> Branch("cl3x5_towerIem",     &_cl3x5_towerIem);
    _tree -> Branch("cl3x5_towerIhad",    &_cl3x5_towerIhad);
    _tree -> Branch("cl3x5_towerIet",     &_cl3x5_towerIet);
    _tree -> Branch("cl3x5_towerEgIet",   &_cl3x5_towerEgIet);
    _tree -> Branch("cl3x5_towerNeg",     &_cl3x5_towerNeg);

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
    iEvent.getByToken(CaloClusters5x7Token, CaloClusters5x7Handle);
    iEvent.getByToken(CaloClusters3x7Token, CaloClusters3x7Handle);
    iEvent.getByToken(CaloClusters3x5Token, CaloClusters3x5Handle);
    iEvent.getByToken(HGClustersToken, HGClustersHandle);
    iEvent.getByToken(genTausToken, genTausHandle);
    iEvent.getByToken(genJetsToken, genJetsHandle);
    
    const TowerHelper::TowerClustersCollection& CaloClusters9x9 = *CaloClusters9x9Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters7x7 = *CaloClusters7x7Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters5x5 = *CaloClusters5x5Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters5x9 = *CaloClusters5x9Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters5x7 = *CaloClusters5x7Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters3x7 = *CaloClusters3x7Handle;
    const TowerHelper::TowerClustersCollection& CaloClusters3x5 = *CaloClusters3x5Handle;
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
        std::cout << " ** total number of 5x7 clusters = " << CaloClusters5x7.size() << std::endl;
        std::cout << " ** total number of 3x7 clusters = " << CaloClusters3x7.size() << std::endl;
        std::cout << " ** total number of 3x5 clusters = " << CaloClusters3x5.size() << std::endl;
        std::cout << " ** total number of hgc clusters = " << HGClusters.size() << std::endl;
        std::cout << "***************************************************************************************************************************************" << std::endl;
    }

    //***************************************************************************************
    //***************************************************************************************
    // PERFORM GEOMETRICAL MATCHING TO TAUS

    for (long unsigned int tauIdx = 0; tauIdx < genTaus.size(); tauIdx++)
    {
        GenHelper::GenTau tau = genTaus[tauIdx];

        if (DEBUG)
        {
            printf(" - GEN TAU pt %f eta %f phi %f vispt %f viseta %f visphi %f DM %i\n",
                tau.pt,
                tau.eta,
                tau.phi,
                tau.visPt,
                tau.visEta,
                tau.visPhi,
                tau.DM);
        }

        // Perform geometrical matching of 9x9 CaloClusters
        int matchedCluIdx = -99;
        float dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters9x9.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu9x9 = CaloClusters9x9[cluIdx];

            if (clu9x9.isPhiFlipped) { continue; }

            float dEta = clu9x9.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu9x9.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 9x9 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu9x9.totalIet,
                    clu9x9.seedEta,
                    clu9x9.seedPhi,
                    clu9x9.totalIem,
                    clu9x9.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu9x9 = const_cast<TowerHelper::TowerCluster&>(CaloClusters9x9[matchedCluIdx]);
            writable_clu9x9.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu9x9_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters9x9[matchedCluIdx+CaloClusters9x9.size()/2]);
            writable_clu9x9_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << " ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 7x7 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters7x7.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu7x7 = CaloClusters7x7[cluIdx];

            if (clu7x7.isPhiFlipped) { continue; }

            float dEta = clu7x7.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu7x7.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 7x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu7x7.totalIet,
                    clu7x7.seedEta,
                    clu7x7.seedPhi,
                    clu7x7.totalIem,
                    clu7x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu7x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters7x7[matchedCluIdx]);
            writable_clu7x7.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu7x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters7x7[matchedCluIdx+CaloClusters7x7.size()/2]);
            writable_clu7x7_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << " ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 5x5 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x5.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu5x5 = CaloClusters5x5[cluIdx];

            if (clu5x5.isPhiFlipped) { continue; }

            float dEta = clu5x5.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu5x5.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x5 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x5.totalIet,
                    clu5x5.seedEta,
                    clu5x5.seedPhi,
                    clu5x5.totalIem,
                    clu5x5.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x5 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x5[matchedCluIdx]);
            writable_clu5x5.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu5x5_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x5[matchedCluIdx+CaloClusters5x5.size()/2]);
            writable_clu5x5_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 5x9 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x9.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu5x9 = CaloClusters5x9[cluIdx];

            if (clu5x9.isPhiFlipped) { continue; }

            float dEta = clu5x9.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu5x9.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x9 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x9.totalIet,
                    clu5x9.seedEta,
                    clu5x9.seedPhi,
                    clu5x9.totalIem,
                    clu5x9.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x9 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x9[matchedCluIdx]);
            writable_clu5x9.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu5x9_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x9[matchedCluIdx+CaloClusters5x9.size()/2]);
            writable_clu5x9_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 5x7 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x7.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu5x7 = CaloClusters5x7[cluIdx];

            if (clu5x7.isPhiFlipped) { continue; }

            float dEta = clu5x7.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu5x7.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x7.totalIet,
                    clu5x7.seedEta,
                    clu5x7.seedPhi,
                    clu5x7.totalIem,
                    clu5x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x7[matchedCluIdx]);
            writable_clu5x7.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu5x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x7[matchedCluIdx+CaloClusters5x7.size()/2]);
            writable_clu5x7_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 3x7 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x7.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu3x7 = CaloClusters3x7[cluIdx];

            if (clu3x7.isPhiFlipped) { continue; }

            float dEta = clu3x7.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu3x7.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 3x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu3x7.totalIet,
                    clu3x7.seedEta,
                    clu3x7.seedPhi,
                    clu3x7.totalIem,
                    clu3x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu3x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x7[matchedCluIdx]);
            writable_clu3x7.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu3x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x7[matchedCluIdx+CaloClusters3x7.size()/2]);
            writable_clu3x7_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 3x5 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x5.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu3x5 = CaloClusters3x5[cluIdx];

            if (clu3x5.isPhiFlipped) { continue; }

            float dEta = clu3x5.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(clu3x5.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 3x5 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu3x5.totalIet,
                    clu3x5.seedEta,
                    clu3x5.seedPhi,
                    clu3x5.totalIem,
                    clu3x5.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu3x5 = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x5[matchedCluIdx]);
            writable_clu3x5.tauMatchIdx = tauIdx;

            TowerHelper::TowerCluster& writable_clu3x5_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x5[matchedCluIdx+CaloClusters3x5.size()/2]);
            writable_clu3x5_phiFlipped.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }
        
        // Perform geometrical matching of HGCAL clusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = hgclu.eta - tau.visEta;
            float dPhi = reco::deltaPhi(hgclu.phi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - HGC CLU pt %f eta %f phi %f e %f dEta %f dPhi %f dR2 %f (%i)\n",
                    hgclu.pt,
                    hgclu.eta,
                    hgclu.phi,
                    hgclu.energy,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            HGClusterHelper::HGCluster& writable_hgclu =  const_cast<HGClusterHelper::HGCluster&>(HGClusters[matchedCluIdx]);
            writable_hgclu.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "\n       *********************************************************************************************************** \n" << std::endl; }

    } // end loop on GenTaus

    if (DEBUG) { std::cout << " ** done with taus, start jets" << std::endl; }

    //***************************************************************************************
    //***************************************************************************************
    // PERFORM GEOMETRICAL MATCHING TO JETS

    for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
    {
        GenHelper::GenJet jet = genJets[jetIdx];

        if (DEBUG)
        {
            printf(" - GEN JET pt %f eta %f phi %f eEM %f eHad %f eInv %f\n",
                jet.pt,
                jet.eta,
                jet.phi,
                jet.eEm,
                jet.eHad,
                jet.eInv);
        }

        // Perform geometrical matching of 9x9 CaloClusters
        int matchedCluIdx = -99;
        float dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters9x9.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu9x9 = CaloClusters9x9[cluIdx];

            if (clu9x9.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu9x9.tauMatchIdx != -99) { continue; }

            float dEta = clu9x9.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu9x9.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 9x9 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu9x9.totalIet,
                    clu9x9.seedEta,
                    clu9x9.seedPhi,
                    clu9x9.totalIem,
                    clu9x9.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu9x9 = const_cast<TowerHelper::TowerCluster&>(CaloClusters9x9[matchedCluIdx]);
            writable_clu9x9.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu9x9_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters9x9[matchedCluIdx+CaloClusters9x9.size()/2]);
            writable_clu9x9_phiFlipped.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 7x7 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters7x7.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu7x7 = CaloClusters7x7[cluIdx];

            if (clu7x7.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu7x7.tauMatchIdx != -99) { continue; }

            float dEta = clu7x7.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu7x7.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 7x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu7x7.totalIet,
                    clu7x7.seedEta,
                    clu7x7.seedPhi,
                    clu7x7.totalIem,
                    clu7x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu7x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters7x7[matchedCluIdx]);
            writable_clu7x7.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu7x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters7x7[matchedCluIdx+CaloClusters7x7.size()/2]);
            writable_clu7x7_phiFlipped.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 5x5 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x5.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu5x5 = CaloClusters5x5[cluIdx];

            if (clu5x5.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu5x5.tauMatchIdx != -99) { continue; }

            float dEta = clu5x5.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu5x5.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x5 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x5.totalIet,
                    clu5x5.seedEta,
                    clu5x5.seedPhi,
                    clu5x5.totalIem,
                    clu5x5.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x5 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x5[matchedCluIdx]);
            writable_clu5x5.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu5x5_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x5[matchedCluIdx+CaloClusters5x5.size()/2]);
            writable_clu5x5_phiFlipped.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 5x9 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x9.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu5x9 = CaloClusters5x9[cluIdx];

            if (clu5x9.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu5x9.tauMatchIdx == -99) { continue; }

            float dEta = clu5x9.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu5x9.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x9 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x9.totalIet,
                    clu5x9.seedEta,
                    clu5x9.seedPhi,
                    clu5x9.totalIem,
                    clu5x9.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x9 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x9[matchedCluIdx]);
            writable_clu5x9.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu5x9_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x9[matchedCluIdx+CaloClusters5x9.size()/2]);
            writable_clu5x9_phiFlipped.jetMatchIdx = jetIdx;
        }

        // Perform geometrical matching of 5x7 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x7.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu5x7 = CaloClusters5x7[cluIdx];

            if (clu5x7.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu5x7.tauMatchIdx == -99) { continue; }

            float dEta = clu5x7.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu5x7.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x7.totalIet,
                    clu5x7.seedEta,
                    clu5x7.seedPhi,
                    clu5x7.totalIem,
                    clu5x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x7[matchedCluIdx]);
            writable_clu5x7.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu5x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x7[matchedCluIdx+CaloClusters5x7.size()/2]);
            writable_clu5x7_phiFlipped.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 3x7 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x7.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu3x7 = CaloClusters3x7[cluIdx];

            if (clu3x7.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu3x7.tauMatchIdx == -99) { continue; }

            float dEta = clu3x7.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu3x7.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 3x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu3x7.totalIet,
                    clu3x7.seedEta,
                    clu3x7.seedPhi,
                    clu3x7.totalIem,
                    clu3x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu3x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x7[matchedCluIdx]);
            writable_clu3x7.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu3x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x7[matchedCluIdx+CaloClusters3x7.size()/2]);
            writable_clu3x7_phiFlipped.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of 3x5 CaloClusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x5.size(); cluIdx++)
        {
            TowerHelper::TowerCluster clu3x5 = CaloClusters3x5[cluIdx];

            if (clu3x5.isPhiFlipped) { continue; }

            // give precedence to gen tau matching
            if (clu3x5.tauMatchIdx == -99) { continue; }

            float dEta = clu3x5.seedEta - jet.eta;
            float dPhi = reco::deltaPhi(clu3x5.seedPhi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - 3x5 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu3x5.totalIet,
                    clu3x5.seedEta,
                    clu3x5.seedPhi,
                    clu3x5.totalIem,
                    clu3x5.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu3x5 = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x5[matchedCluIdx]);
            writable_clu3x5.jetMatchIdx = jetIdx;

            TowerHelper::TowerCluster& writable_clu3x5_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x5[matchedCluIdx+CaloClusters3x5.size()/2]);
            writable_clu3x5_phiFlipped.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }

        // Perform geometrical matching of HGCAL clusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            // give precedence to gen tau matching
            if (hgclu.tauMatchIdx == -99) { continue; }

            float dEta = hgclu.eta - jet.eta;
            float dPhi = reco::deltaPhi(hgclu.phi, jet.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - HGC CLU pt %f eta %f phi %f e %f dEta %f dPhi %f dR2 %f (%i)\n",
                    hgclu.pt,
                    hgclu.eta,
                    hgclu.phi,
                    hgclu.energy,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            HGClusterHelper::HGCluster& writable_hgclu = const_cast<HGClusterHelper::HGCluster&>(HGClusters[matchedCluIdx]);
            writable_hgclu.jetMatchIdx = jetIdx;
        }

        if (DEBUG) { std::cout << "\n       *********************************************************************************************************** \n" << std::endl; }

    } // end loop on GenJets

    //***************************************************************************************
    //***************************************************************************************
    // PERFORM GEOMETRICAL MATCHING BETWEEN CL3D AND CLTW IN THE ENDCAP

    // Perform geometrical matching of 9x9 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters9x9.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu9x9 = CaloClusters9x9[cluIdx];

        if (clu9x9.isPhiFlipped) { continue; }
        
        if (abs(clu9x9.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu9x9.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu9x9.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 9x9 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu9x9.totalIet,
                    clu9x9.seedEta,
                    clu9x9.seedPhi,
                    clu9x9.totalIem,
                    clu9x9.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu9x9 = const_cast<TowerHelper::TowerCluster&>(CaloClusters9x9[cluIdx]);
            writable_clu9x9.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu9x9_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters9x9[cluIdx+CaloClusters9x9.size()/2]);
            writable_clu9x9_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
    }

    // Perform geometrical matching of 7x7 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters7x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu7x7 = CaloClusters7x7[cluIdx];

        if (clu7x7.isPhiFlipped) { continue; }

        if (abs(clu7x7.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu7x7.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu7x7.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 7x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu7x7.totalIet,
                    clu7x7.seedEta,
                    clu7x7.seedPhi,
                    clu7x7.totalIem,
                    clu7x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu7x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters7x7[cluIdx]);
            writable_clu7x7.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu7x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters7x7[cluIdx+CaloClusters7x7.size()/2]);
            writable_clu7x7_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
    }

    // Perform geometrical matching of 5x5 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x5.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x5 = CaloClusters5x5[cluIdx];

        if (clu5x5.isPhiFlipped) { continue; }

        if (abs(clu5x5.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu5x5.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu5x5.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x5 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x5.totalIet,
                    clu5x5.seedEta,
                    clu5x5.seedPhi,
                    clu5x5.totalIem,
                    clu5x5.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x5 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x5[cluIdx]);
            writable_clu5x5.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu5x5_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x5[cluIdx+CaloClusters5x5.size()/2]);
            writable_clu5x5_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
    }

    // Perform geometrical matching of 5x9 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x9.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x9 = CaloClusters5x9[cluIdx];

        if (clu5x9.isPhiFlipped) { continue; }

        if (abs(clu5x9.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu5x9.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu5x9.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x9 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x9.totalIet,
                    clu5x9.seedEta,
                    clu5x9.seedPhi,
                    clu5x9.totalIem,
                    clu5x9.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x9 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x9[cluIdx]);
            writable_clu5x9.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu5x9_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x9[cluIdx+CaloClusters5x9.size()/2]);
            writable_clu5x9_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
    }

    // Perform geometrical matching of 5x7 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x7 = CaloClusters5x7[cluIdx];

        if (clu5x7.isPhiFlipped) { continue; }

        if (abs(clu5x7.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu5x7.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu5x7.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 5x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu5x7.totalIet,
                    clu5x7.seedEta,
                    clu5x7.seedPhi,
                    clu5x7.totalIem,
                    clu5x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu5x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x7[cluIdx]);
            writable_clu5x7.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu5x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters5x7[cluIdx+CaloClusters5x7.size()/2]);
            writable_clu5x7_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
    }

    // Perform geometrical matching of 3x7 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu3x7 = CaloClusters3x7[cluIdx];

        if (clu3x7.isPhiFlipped) { continue; }

        if (abs(clu3x7.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu3x7.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu3x7.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 3x7 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu3x7.totalIet,
                    clu3x7.seedEta,
                    clu3x7.seedPhi,
                    clu3x7.totalIem,
                    clu3x7.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu3x7 = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x7[cluIdx]);
            writable_clu3x7.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu3x7_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x7[cluIdx+CaloClusters3x7.size()/2]);
            writable_clu3x7_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
    }

    // Perform geometrical matching of 3x5 CaloClusters
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x5.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu3x5 = CaloClusters3x5[cluIdx];

        if (clu3x5.isPhiFlipped) { continue; }

        if (abs(clu3x5.seedEta) < 1.5) { continue; } // skip the cltw that by construction cannot be matched to cl3d

        int matchedHGCluIdx = -99;
        float ptMax = -99.;
        for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
        {
            HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

            float dEta = clu3x5.seedEta - hgclu.eta;
            float dPhi = reco::deltaPhi(clu3x5.seedPhi, hgclu.phi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= 0.25 && hgclu.pt > ptMax)
            {
                ptMax = hgclu.pt;
                matchedHGCluIdx = hgcluIdx;
            }

            if (DEBUG)
            {
                printf("         - 3x5 TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    clu3x5.totalIet,
                    clu3x5.seedEta,
                    clu3x5.seedPhi,
                    clu3x5.totalIem,
                    clu3x5.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedHGCluIdx);
            }
        }
        if (matchedHGCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_clu3x5 = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x5[cluIdx]);
            writable_clu3x5.cl3dMatchIdx = matchedHGCluIdx;

            TowerHelper::TowerCluster& writable_clu3x5_phiFlipped = const_cast<TowerHelper::TowerCluster&>(CaloClusters3x5[cluIdx+CaloClusters3x5.size()/2]);
            writable_clu3x5_phiFlipped.cl3dMatchIdx = matchedHGCluIdx;
        }
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

    // Fill GenJet branches
    for (long unsigned int jetIdx = 0; jetIdx < genJets.size(); jetIdx++)
    {
        GenHelper::GenJet jet = genJets[jetIdx];

        _jet_eta.push_back(jet.eta);
        _jet_phi.push_back(jet.phi);
        _jet_pt.push_back(jet.pt);
        _jet_e.push_back(jet.e);
        _jet_eEm.push_back(jet.eEm);
        _jet_eHad.push_back(jet.eHad);
        _jet_eInv.push_back(jet.eInv);
        _jet_Idx.push_back(jetIdx);
    }

    // Fill 9x9 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters9x9.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu9x9 = CaloClusters9x9[cluIdx];

        _cl9x9_barrelSeeded.push_back(clu9x9.barrelSeeded);
        _cl9x9_nHits.push_back(clu9x9.nHits);
        _cl9x9_seedIeta.push_back(clu9x9.seedIeta);
        _cl9x9_seedIphi.push_back(clu9x9.seedIphi);
        _cl9x9_seedEta.push_back(clu9x9.seedEta);
        _cl9x9_seedPhi.push_back(clu9x9.seedPhi);
        _cl9x9_isBarrel.push_back(clu9x9.isBarrel);
        _cl9x9_isOverlap.push_back(clu9x9.isOverlap);
        _cl9x9_isEndcap.push_back(clu9x9.isEndcap);
        _cl9x9_isPhiFlipped.push_back(clu9x9.isPhiFlipped);
        _cl9x9_tauMatchIdx.push_back(clu9x9.tauMatchIdx);
        _cl9x9_jetMatchIdx.push_back(clu9x9.jetMatchIdx);
        _cl9x9_cl3dMatchIdx.push_back(clu9x9.cl3dMatchIdx);
        _cl9x9_totalEm.push_back(clu9x9.totalEm);
        _cl9x9_totalHad.push_back(clu9x9.totalHad);
        _cl9x9_totalEt.push_back(clu9x9.totalEt);
        _cl9x9_totalEgEt.push_back(clu9x9.totalEgEt);
        _cl9x9_totalIem.push_back(clu9x9.totalIem);
        _cl9x9_totalIhad.push_back(clu9x9.totalIhad);
        _cl9x9_totalIet.push_back(clu9x9.totalIet);
        _cl9x9_totalEgIet.push_back(clu9x9.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu9x9.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu9x9.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu9x9.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu9x9.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu9x9.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu9x9.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu9x9.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu9x9.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu9x9.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu9x9.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu9x9.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu9x9.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu9x9.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu9x9.towerHits[i].nL1eg);

            nEGs += clu9x9.towerHits[i].nL1eg;
        }

        _cl9x9_towerEta.push_back(tmp_towerEta);
        _cl9x9_towerPhi.push_back(tmp_towerPhi);
        _cl9x9_towerEm.push_back(tmp_towerEm);
        _cl9x9_towerHad.push_back(tmp_towerHad);
        _cl9x9_towerEt.push_back(tmp_towerEt);
        _cl9x9_towerEgEt.push_back(tmp_towerEgEt);
        _cl9x9_towerIeta.push_back(tmp_towerIeta);
        _cl9x9_towerIphi.push_back(tmp_towerIphi);
        _cl9x9_towerIem.push_back(tmp_towerIem);
        _cl9x9_towerIhad.push_back(tmp_towerIhad);
        _cl9x9_towerIet.push_back(tmp_towerIet);
        _cl9x9_towerEgIet.push_back(tmp_towerEgIet);
        _cl9x9_towerNeg.push_back(tmp_towerNeg);

        _cl9x9_nEGs.push_back(nEGs);
    }

    // Fill 7x7 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters7x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu7x7 = CaloClusters7x7[cluIdx];

        _cl7x7_barrelSeeded.push_back(clu7x7.barrelSeeded);
        _cl7x7_nHits.push_back(clu7x7.nHits);
        _cl7x7_seedIeta.push_back(clu7x7.seedIeta);
        _cl7x7_seedIphi.push_back(clu7x7.seedIphi);
        _cl7x7_seedEta.push_back(clu7x7.seedEta);
        _cl7x7_seedPhi.push_back(clu7x7.seedPhi);
        _cl7x7_isBarrel.push_back(clu7x7.isBarrel);
        _cl7x7_isOverlap.push_back(clu7x7.isOverlap);
        _cl7x7_isEndcap.push_back(clu7x7.isEndcap);
        _cl7x7_isPhiFlipped.push_back(clu7x7.isPhiFlipped);
        _cl7x7_tauMatchIdx.push_back(clu7x7.tauMatchIdx);
        _cl7x7_jetMatchIdx.push_back(clu7x7.jetMatchIdx);
        _cl7x7_cl3dMatchIdx.push_back(clu7x7.cl3dMatchIdx);
        _cl7x7_totalEm.push_back(clu7x7.totalEm);
        _cl7x7_totalHad.push_back(clu7x7.totalHad);
        _cl7x7_totalEt.push_back(clu7x7.totalEt);
        _cl7x7_totalEgEt.push_back(clu7x7.totalEgEt);
        _cl7x7_totalIem.push_back(clu7x7.totalIem);
        _cl7x7_totalIhad.push_back(clu7x7.totalIhad);
        _cl7x7_totalIet.push_back(clu7x7.totalIet);
        _cl7x7_totalEgIet.push_back(clu7x7.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu7x7.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu7x7.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu7x7.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu7x7.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu7x7.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu7x7.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu7x7.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu7x7.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu7x7.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu7x7.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu7x7.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu7x7.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu7x7.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu7x7.towerHits[i].nL1eg);

            nEGs += clu7x7.towerHits[i].nL1eg;
        }

        _cl7x7_towerEta.push_back(tmp_towerEta);
        _cl7x7_towerPhi.push_back(tmp_towerPhi);
        _cl7x7_towerEm.push_back(tmp_towerEm);
        _cl7x7_towerHad.push_back(tmp_towerHad);
        _cl7x7_towerEt.push_back(tmp_towerEt);
        _cl7x7_towerEgEt.push_back(tmp_towerEgEt);
        _cl7x7_towerIeta.push_back(tmp_towerIeta);
        _cl7x7_towerIphi.push_back(tmp_towerIphi);
        _cl7x7_towerIem.push_back(tmp_towerIem);
        _cl7x7_towerIhad.push_back(tmp_towerIhad);
        _cl7x7_towerIet.push_back(tmp_towerIet);
        _cl7x7_towerEgIet.push_back(tmp_towerEgIet);
        _cl7x7_towerNeg.push_back(tmp_towerNeg);

        _cl7x7_nEGs.push_back(nEGs);
    }

    // Fill 5x5 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x5.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x5 = CaloClusters5x5[cluIdx];

        _cl5x5_barrelSeeded.push_back(clu5x5.barrelSeeded);
        _cl5x5_nHits.push_back(clu5x5.nHits);
        _cl5x5_seedIeta.push_back(clu5x5.seedIeta);
        _cl5x5_seedIphi.push_back(clu5x5.seedIphi);
        _cl5x5_seedEta.push_back(clu5x5.seedEta);
        _cl5x5_seedPhi.push_back(clu5x5.seedPhi);
        _cl5x5_isBarrel.push_back(clu5x5.isBarrel);
        _cl5x5_isOverlap.push_back(clu5x5.isOverlap);
        _cl5x5_isEndcap.push_back(clu5x5.isEndcap);
        _cl5x5_isPhiFlipped.push_back(clu5x5.isPhiFlipped);
        _cl5x5_tauMatchIdx.push_back(clu5x5.tauMatchIdx);
        _cl5x5_jetMatchIdx.push_back(clu5x5.jetMatchIdx);
        _cl5x5_cl3dMatchIdx.push_back(clu5x5.cl3dMatchIdx);
        _cl5x5_totalEm.push_back(clu5x5.totalEm);
        _cl5x5_totalHad.push_back(clu5x5.totalHad);
        _cl5x5_totalEt.push_back(clu5x5.totalEt);
        _cl5x5_totalEgEt.push_back(clu5x5.totalEgEt);
        _cl5x5_totalIem.push_back(clu5x5.totalIem);
        _cl5x5_totalIhad.push_back(clu5x5.totalIhad);
        _cl5x5_totalIet.push_back(clu5x5.totalIet);
        _cl5x5_totalEgIet.push_back(clu5x5.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu5x5.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu5x5.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu5x5.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu5x5.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu5x5.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu5x5.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu5x5.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu5x5.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu5x5.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu5x5.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu5x5.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu5x5.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu5x5.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu5x5.towerHits[i].nL1eg);

            nEGs += clu5x5.towerHits[i].nL1eg;
        }

        _cl5x5_towerEta.push_back(tmp_towerEta);
        _cl5x5_towerPhi.push_back(tmp_towerPhi);
        _cl5x5_towerEm.push_back(tmp_towerEm);
        _cl5x5_towerHad.push_back(tmp_towerHad);
        _cl5x5_towerEt.push_back(tmp_towerEt);
        _cl5x5_towerEgEt.push_back(tmp_towerEgEt);
        _cl5x5_towerIeta.push_back(tmp_towerIeta);
        _cl5x5_towerIphi.push_back(tmp_towerIphi);
        _cl5x5_towerIem.push_back(tmp_towerIem);
        _cl5x5_towerIhad.push_back(tmp_towerIhad);
        _cl5x5_towerIet.push_back(tmp_towerIet);
        _cl5x5_towerEgIet.push_back(tmp_towerEgIet);
        _cl5x5_towerNeg.push_back(tmp_towerNeg);

        _cl5x5_nEGs.push_back(nEGs);
    }

    // Fill 5x9 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x9.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x9 = CaloClusters5x9[cluIdx];

        _cl5x9_barrelSeeded.push_back(clu5x9.barrelSeeded);
        _cl5x9_nHits.push_back(clu5x9.nHits);
        _cl5x9_seedIeta.push_back(clu5x9.seedIeta);
        _cl5x9_seedIphi.push_back(clu5x9.seedIphi);
        _cl5x9_seedEta.push_back(clu5x9.seedEta);
        _cl5x9_seedPhi.push_back(clu5x9.seedPhi);
        _cl5x9_isBarrel.push_back(clu5x9.isBarrel);
        _cl5x9_isOverlap.push_back(clu5x9.isOverlap);
        _cl5x9_isEndcap.push_back(clu5x9.isEndcap);
        _cl5x9_isPhiFlipped.push_back(clu5x9.isPhiFlipped);
        _cl5x9_tauMatchIdx.push_back(clu5x9.tauMatchIdx);
        _cl5x9_jetMatchIdx.push_back(clu5x9.jetMatchIdx);
        _cl5x9_cl3dMatchIdx.push_back(clu5x9.cl3dMatchIdx);
        _cl5x9_totalEm.push_back(clu5x9.totalEm);
        _cl5x9_totalHad.push_back(clu5x9.totalHad);
        _cl5x9_totalEt.push_back(clu5x9.totalEt);
        _cl5x9_totalEgEt.push_back(clu5x9.totalEgEt);
        _cl5x9_totalIem.push_back(clu5x9.totalIem);
        _cl5x9_totalIhad.push_back(clu5x9.totalIhad);
        _cl5x9_totalIet.push_back(clu5x9.totalIet);
        _cl5x9_totalEgIet.push_back(clu5x9.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu5x9.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu5x9.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu5x9.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu5x9.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu5x9.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu5x9.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu5x9.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu5x9.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu5x9.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu5x9.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu5x9.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu5x9.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu5x9.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu5x9.towerHits[i].nL1eg);

            nEGs += clu5x9.towerHits[i].nL1eg;
        }

        _cl5x9_towerEta.push_back(tmp_towerEta);
        _cl5x9_towerPhi.push_back(tmp_towerPhi);
        _cl5x9_towerEm.push_back(tmp_towerEm);
        _cl5x9_towerHad.push_back(tmp_towerHad);
        _cl5x9_towerEt.push_back(tmp_towerEt);
        _cl5x9_towerEgEt.push_back(tmp_towerEgEt);
        _cl5x9_towerIeta.push_back(tmp_towerIeta);
        _cl5x9_towerIphi.push_back(tmp_towerIphi);
        _cl5x9_towerIem.push_back(tmp_towerIem);
        _cl5x9_towerIhad.push_back(tmp_towerIhad);
        _cl5x9_towerIet.push_back(tmp_towerIet);
        _cl5x9_towerEgIet.push_back(tmp_towerEgIet);
        _cl5x9_towerNeg.push_back(tmp_towerNeg);

        _cl5x9_nEGs.push_back(nEGs);
    }

    // Fill 5x7 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters5x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu5x7 = CaloClusters5x7[cluIdx];

        _cl5x7_barrelSeeded.push_back(clu5x7.barrelSeeded);
        _cl5x7_nHits.push_back(clu5x7.nHits);
        _cl5x7_seedIeta.push_back(clu5x7.seedIeta);
        _cl5x7_seedIphi.push_back(clu5x7.seedIphi);
        _cl5x7_seedEta.push_back(clu5x7.seedEta);
        _cl5x7_seedPhi.push_back(clu5x7.seedPhi);
        _cl5x7_isBarrel.push_back(clu5x7.isBarrel);
        _cl5x7_isOverlap.push_back(clu5x7.isOverlap);
        _cl5x7_isEndcap.push_back(clu5x7.isEndcap);
        _cl5x7_isPhiFlipped.push_back(clu5x7.isPhiFlipped);
        _cl5x7_tauMatchIdx.push_back(clu5x7.tauMatchIdx);
        _cl5x7_jetMatchIdx.push_back(clu5x7.jetMatchIdx);
        _cl5x7_cl3dMatchIdx.push_back(clu5x7.cl3dMatchIdx);
        _cl5x7_totalEm.push_back(clu5x7.totalEm);
        _cl5x7_totalHad.push_back(clu5x7.totalHad);
        _cl5x7_totalEt.push_back(clu5x7.totalEt);
        _cl5x7_totalEgEt.push_back(clu5x7.totalEgEt);
        _cl5x7_totalIem.push_back(clu5x7.totalIem);
        _cl5x7_totalIhad.push_back(clu5x7.totalIhad);
        _cl5x7_totalIet.push_back(clu5x7.totalIet);
        _cl5x7_totalEgIet.push_back(clu5x7.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu5x7.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu5x7.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu5x7.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu5x7.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu5x7.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu5x7.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu5x7.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu5x7.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu5x7.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu5x7.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu5x7.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu5x7.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu5x7.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu5x7.towerHits[i].nL1eg);

            nEGs += clu5x7.towerHits[i].nL1eg;
        }

        _cl5x7_towerEta.push_back(tmp_towerEta);
        _cl5x7_towerPhi.push_back(tmp_towerPhi);
        _cl5x7_towerEm.push_back(tmp_towerEm);
        _cl5x7_towerHad.push_back(tmp_towerHad);
        _cl5x7_towerEt.push_back(tmp_towerEt);
        _cl5x7_towerEgEt.push_back(tmp_towerEgEt);
        _cl5x7_towerIeta.push_back(tmp_towerIeta);
        _cl5x7_towerIphi.push_back(tmp_towerIphi);
        _cl5x7_towerIem.push_back(tmp_towerIem);
        _cl5x7_towerIhad.push_back(tmp_towerIhad);
        _cl5x7_towerIet.push_back(tmp_towerIet);
        _cl5x7_towerEgIet.push_back(tmp_towerEgIet);
        _cl5x7_towerNeg.push_back(tmp_towerNeg);

        _cl5x7_nEGs.push_back(nEGs);
    }

    // Fill 3x7 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x7.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu3x7 = CaloClusters3x7[cluIdx];

        _cl3x7_barrelSeeded.push_back(clu3x7.barrelSeeded);
        _cl3x7_nHits.push_back(clu3x7.nHits);
        _cl3x7_seedIeta.push_back(clu3x7.seedIeta);
        _cl3x7_seedIphi.push_back(clu3x7.seedIphi);
        _cl3x7_seedEta.push_back(clu3x7.seedEta);
        _cl3x7_seedPhi.push_back(clu3x7.seedPhi);
        _cl3x7_isBarrel.push_back(clu3x7.isBarrel);
        _cl3x7_isOverlap.push_back(clu3x7.isOverlap);
        _cl3x7_isEndcap.push_back(clu3x7.isEndcap);
        _cl3x7_isPhiFlipped.push_back(clu3x7.isPhiFlipped);
        _cl3x7_tauMatchIdx.push_back(clu3x7.tauMatchIdx);
        _cl3x7_jetMatchIdx.push_back(clu3x7.jetMatchIdx);
        _cl3x7_cl3dMatchIdx.push_back(clu3x7.cl3dMatchIdx);
        _cl3x7_totalEm.push_back(clu3x7.totalEm);
        _cl3x7_totalHad.push_back(clu3x7.totalHad);
        _cl3x7_totalEt.push_back(clu3x7.totalEt);
        _cl3x7_totalEgEt.push_back(clu3x7.totalEgEt);
        _cl3x7_totalIem.push_back(clu3x7.totalIem);
        _cl3x7_totalIhad.push_back(clu3x7.totalIhad);
        _cl3x7_totalIet.push_back(clu3x7.totalIet);
        _cl3x7_totalEgIet.push_back(clu3x7.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu3x7.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu3x7.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu3x7.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu3x7.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu3x7.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu3x7.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu3x7.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu3x7.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu3x7.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu3x7.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu3x7.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu3x7.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu3x7.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu3x7.towerHits[i].nL1eg);

            nEGs += clu3x7.towerHits[i].nL1eg;
        }

        _cl3x7_towerEta.push_back(tmp_towerEta);
        _cl3x7_towerPhi.push_back(tmp_towerPhi);
        _cl3x7_towerEm.push_back(tmp_towerEm);
        _cl3x7_towerHad.push_back(tmp_towerHad);
        _cl3x7_towerEt.push_back(tmp_towerEt);
        _cl3x7_towerEgEt.push_back(tmp_towerEgEt);
        _cl3x7_towerIeta.push_back(tmp_towerIeta);
        _cl3x7_towerIphi.push_back(tmp_towerIphi);
        _cl3x7_towerIem.push_back(tmp_towerIem);
        _cl3x7_towerIhad.push_back(tmp_towerIhad);
        _cl3x7_towerIet.push_back(tmp_towerIet);
        _cl3x7_towerEgIet.push_back(tmp_towerEgIet);
        _cl3x7_towerNeg.push_back(tmp_towerNeg);

        _cl3x7_nEGs.push_back(nEGs);
    }

    // Fill 3x5 CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClusters3x5.size(); cluIdx++)
    {
        TowerHelper::TowerCluster clu3x5 = CaloClusters3x5[cluIdx];

        _cl3x5_barrelSeeded.push_back(clu3x5.barrelSeeded);
        _cl3x5_nHits.push_back(clu3x5.nHits);
        _cl3x5_seedIeta.push_back(clu3x5.seedIeta);
        _cl3x5_seedIphi.push_back(clu3x5.seedIphi);
        _cl3x5_seedEta.push_back(clu3x5.seedEta);
        _cl3x5_seedPhi.push_back(clu3x5.seedPhi);
        _cl3x5_isBarrel.push_back(clu3x5.isBarrel);
        _cl3x5_isOverlap.push_back(clu3x5.isOverlap);
        _cl3x5_isEndcap.push_back(clu3x5.isEndcap);
        _cl3x5_isPhiFlipped.push_back(clu3x5.isPhiFlipped);
        _cl3x5_tauMatchIdx.push_back(clu3x5.tauMatchIdx);
        _cl3x5_jetMatchIdx.push_back(clu3x5.jetMatchIdx);
        _cl3x5_cl3dMatchIdx.push_back(clu3x5.cl3dMatchIdx);
        _cl3x5_totalEm.push_back(clu3x5.totalEm);
        _cl3x5_totalHad.push_back(clu3x5.totalHad);
        _cl3x5_totalEt.push_back(clu3x5.totalEt);
        _cl3x5_totalEgEt.push_back(clu3x5.totalEgEt);
        _cl3x5_totalIem.push_back(clu3x5.totalIem);
        _cl3x5_totalIhad.push_back(clu3x5.totalIhad);
        _cl3x5_totalIet.push_back(clu3x5.totalIet);
        _cl3x5_totalEgIet.push_back(clu3x5.totalEgIet);
        
        std::vector<float> tmp_towerEta;
        std::vector<float> tmp_towerPhi;
        std::vector<float> tmp_towerEm;
        std::vector<float> tmp_towerHad;
        std::vector<float> tmp_towerEt;
        std::vector<float> tmp_towerEgEt;
        std::vector<int>   tmp_towerIeta;
        std::vector<int>   tmp_towerIphi;
        std::vector<int>   tmp_towerIem;
        std::vector<int>   tmp_towerIhad;
        std::vector<int>   tmp_towerIet;
        std::vector<int>   tmp_towerEgIet;
        std::vector<int>   tmp_towerNeg;
        int nEGs = 0;

        for (long unsigned int i = 0; i < clu3x5.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(clu3x5.towerHits[i].towerEta);
            tmp_towerPhi.push_back(clu3x5.towerHits[i].towerPhi);
            tmp_towerEm.push_back(clu3x5.towerHits[i].towerEm);
            tmp_towerHad.push_back(clu3x5.towerHits[i].towerHad);
            tmp_towerEt.push_back(clu3x5.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(clu3x5.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(clu3x5.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(clu3x5.towerHits[i].towerIphi);
            tmp_towerIem.push_back(clu3x5.towerHits[i].towerIem);
            tmp_towerIhad.push_back(clu3x5.towerHits[i].towerIhad);
            tmp_towerIet.push_back(clu3x5.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(clu3x5.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(clu3x5.towerHits[i].nL1eg);

            nEGs += clu3x5.towerHits[i].nL1eg;
        }

        _cl3x5_towerEta.push_back(tmp_towerEta);
        _cl3x5_towerPhi.push_back(tmp_towerPhi);
        _cl3x5_towerEm.push_back(tmp_towerEm);
        _cl3x5_towerHad.push_back(tmp_towerHad);
        _cl3x5_towerEt.push_back(tmp_towerEt);
        _cl3x5_towerEgEt.push_back(tmp_towerEgEt);
        _cl3x5_towerIeta.push_back(tmp_towerIeta);
        _cl3x5_towerIphi.push_back(tmp_towerIphi);
        _cl3x5_towerIem.push_back(tmp_towerIem);
        _cl3x5_towerIhad.push_back(tmp_towerIhad);
        _cl3x5_towerIet.push_back(tmp_towerIet);
        _cl3x5_towerEgIet.push_back(tmp_towerEgIet);
        _cl3x5_towerNeg.push_back(tmp_towerNeg);

        _cl3x5_nEGs.push_back(nEGs);
    }

    // Fill HGCluster branches
    for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
    {
        HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

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
        _cl3d_puid.push_back(hgclu.puId);
        _cl3d_puidscore.push_back(hgclu.puIdScore);
        _cl3d_pionid.push_back(hgclu.pionId);
        _cl3d_pionidscore.push_back(hgclu.pionIdScore);
        _cl3d_tauMatchIdx.push_back(hgclu.tauMatchIdx);
        _cl3d_jetMatchIdx.push_back(hgclu.jetMatchIdx);
    }

    if (DEBUG) { std::cout << " ** finished macthing, now filling the tree for run " << _runNumber << " - event " << _evtNumber << std::endl; }

    // Fill tree
    _tree -> Fill();
}

DEFINE_FWK_MODULE(Ntuplizer);