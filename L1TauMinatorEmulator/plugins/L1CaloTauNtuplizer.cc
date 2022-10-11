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
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
#include "L1TauMinator/DataFormats/interface/GenHelper.h"
#include "L1TauMinator/DataFormats/interface/TauHelper.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


// class L1CaloTauNtuplizer : public edm::stream::EDAnalyzer<> {
class L1CaloTauNtuplizer : public edm::EDAnalyzer {
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
        edm::EDGetTokenT<TowerHelper::TowerClustersCollection> CaloClustersNxMToken;
        edm::Handle<TowerHelper::TowerClustersCollection> CaloClustersNxMHandle;

        edm::EDGetTokenT<HGClusterHelper::HGClustersCollection> HGClustersToken;
        edm::Handle<HGClusterHelper::HGClustersCollection> HGClustersHandle;

        edm::EDGetTokenT<TauHelper::TausCollection> minatedTausToken;
        edm::Handle<TauHelper::TausCollection> minatedTausHandle;

        edm::EDGetTokenT<l1t::TauBxCollection> squareTausToken;
        edm::Handle<BXVector<l1t::Tau>>  squareTausHandle;

        edm::EDGetTokenT<GenHelper::GenTausCollection> genTausToken;
        edm::Handle<GenHelper::GenTausCollection> genTausHandle;

        //----private variables----
        int etaClusterDimension;
        int phiClusterDimension;

        bool DEBUG;

        TTree *_tree;
        TTree *_triggerNamesTree;
        std::string _treeName;
        std::string _NxM;

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

        std::vector<float>  _minatedl1tau_pt;
        std::vector<float>  _minatedl1tau_eta;
        std::vector<float>  _minatedl1tau_phi;
        std::vector<int>    _minatedl1tau_clusterIdx;
        std::vector<bool>   _minatedl1tau_isBarrel;
        std::vector<bool>   _minatedl1tau_isEndcap;
        std::vector<float>  _minatedl1tau_IDscore;
        std::vector<int>    _minatedl1tau_tauMatchIdx;

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

        
        std::vector<float> _clNxM_calibPt;
        std::vector<float> _clNxM_IDscore;
        std::vector<bool>  _clNxM_barrelSeeded;
        std::vector<int>   _clNxM_nHits;
        std::vector<int>   _clNxM_nEGs;
        std::vector<int>   _clNxM_seedIeta;
        std::vector<int>   _clNxM_seedIphi;
        std::vector<float> _clNxM_seedEta;
        std::vector<float> _clNxM_seedPhi;
        std::vector<bool>  _clNxM_isBarrel;
        std::vector<bool>  _clNxM_isOverlap;
        std::vector<bool>  _clNxM_isEndcap;
        std::vector<int>   _clNxM_tauMatchIdx;
        std::vector<int>   _clNxM_jetMatchIdx;
        std::vector<float> _clNxM_totalEm;
        std::vector<float> _clNxM_totalHad;
        std::vector<float> _clNxM_totalEt;
        std::vector<float> _clNxM_totalEgEt;
        std::vector<int>   _clNxM_totalIem;
        std::vector<int>   _clNxM_totalIhad;
        std::vector<int>   _clNxM_totalIet;
        std::vector<int>   _clNxM_totalEgIet;
        std::vector< std::vector<float> > _clNxM_towerEta;
        std::vector< std::vector<float> > _clNxM_towerPhi;
        std::vector< std::vector<float> > _clNxM_towerEm;
        std::vector< std::vector<float> > _clNxM_towerHad;
        std::vector< std::vector<float> > _clNxM_towerEt;
        std::vector< std::vector<float> > _clNxM_towerEgEt;
        std::vector< std::vector<int> >   _clNxM_towerIeta;
        std::vector< std::vector<int> >   _clNxM_towerIphi;
        std::vector< std::vector<int> >   _clNxM_towerIem;
        std::vector< std::vector<int> >   _clNxM_towerIhad;
        std::vector< std::vector<int> >   _clNxM_towerIet;
        std::vector< std::vector<int> >   _clNxM_towerEgIet;
        std::vector< std::vector<int> >   _clNxM_towerNeg;
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
    : CaloClustersNxMToken(consumes<TowerHelper::TowerClustersCollection>(iConfig.getParameter<edm::InputTag>("CaloClustersNxM"))),
      HGClustersToken(consumes<HGClusterHelper::HGClustersCollection>(iConfig.getParameter<edm::InputTag>("HGClusters"))),
      minatedTausToken(consumes<TauHelper::TausCollection>(iConfig.getParameter<edm::InputTag>("minatedTaus"))),
      squareTausToken(consumes<l1t::TauBxCollection>(iConfig.getParameter<edm::InputTag>("squareTaus"))),
      genTausToken(consumes<GenHelper::GenTausCollection>(iConfig.getParameter<edm::InputTag>("genTaus"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{
    _treeName = iConfig.getParameter<std::string>("treeName");
    etaClusterDimension = iConfig.getParameter<int>("etaClusterDimension"),
    phiClusterDimension = iConfig.getParameter<int>("phiClusterDimension"),
    _NxM = std::to_string(etaClusterDimension)+"x"+std::to_string(phiClusterDimension);
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

    _minatedl1tau_pt.clear();
    _minatedl1tau_eta.clear();
    _minatedl1tau_phi.clear();
    _minatedl1tau_clusterIdx.clear();
    _minatedl1tau_isBarrel.clear();
    _minatedl1tau_isEndcap.clear();
    _minatedl1tau_IDscore.clear();
    _minatedl1tau_tauMatchIdx.clear();

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

    _clNxM_calibPt.clear();
    _clNxM_IDscore.clear();
    _clNxM_barrelSeeded.clear();
    _clNxM_nHits.clear();
    _clNxM_nEGs.clear();
    _clNxM_seedIeta.clear();
    _clNxM_seedIphi.clear();
    _clNxM_seedEta.clear();
    _clNxM_seedPhi.clear();
    _clNxM_isBarrel.clear();
    _clNxM_isOverlap.clear();
    _clNxM_isEndcap.clear();
    _clNxM_tauMatchIdx.clear();
    _clNxM_jetMatchIdx.clear();
    _clNxM_totalEm.clear();
    _clNxM_totalHad.clear();
    _clNxM_totalEt.clear();
    _clNxM_totalEgEt.clear();
    _clNxM_totalIem.clear();
    _clNxM_totalIhad.clear();
    _clNxM_totalIet.clear();
    _clNxM_totalEgIet.clear();
    _clNxM_towerEta.clear();
    _clNxM_towerPhi.clear();
    _clNxM_towerEm.clear();
    _clNxM_towerHad.clear();
    _clNxM_towerEt.clear();
    _clNxM_towerEgEt.clear();
    _clNxM_towerIeta.clear();
    _clNxM_towerIphi.clear();
    _clNxM_towerIem.clear();
    _clNxM_towerIhad.clear();
    _clNxM_towerIet.clear();
    _clNxM_towerEgIet.clear();
    _clNxM_towerNeg.clear();
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

    _tree -> Branch("minatedl1tau_pt",          &_minatedl1tau_pt);
    _tree -> Branch("minatedl1tau_eta",         &_minatedl1tau_eta);
    _tree -> Branch("minatedl1tau_phi",         &_minatedl1tau_phi);
    _tree -> Branch("minatedl1tau_clusterIdx",  &_minatedl1tau_clusterIdx);
    _tree -> Branch("minatedl1tau_isBarrel",    &_minatedl1tau_isBarrel);
    _tree -> Branch("minatedl1tau_isEndcap",    &_minatedl1tau_isEndcap);
    _tree -> Branch("minatedl1tau_IDscore",     &_minatedl1tau_IDscore);
    _tree -> Branch("minatedl1tau_tauMatchIdx", &_minatedl1tau_tauMatchIdx);

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

    _tree -> Branch( ("cl"+_NxM+"_calibPt").c_str(),      &_clNxM_calibPt);
    _tree -> Branch( ("cl"+_NxM+"_IDscore").c_str(),      &_clNxM_IDscore);
    _tree -> Branch( ("cl"+_NxM+"_barrelSeeded").c_str(), &_clNxM_barrelSeeded);
    _tree -> Branch( ("cl"+_NxM+"_nHits").c_str(),        &_clNxM_nHits);
    _tree -> Branch( ("cl"+_NxM+"_nEGs").c_str(),         &_clNxM_nEGs);
    _tree -> Branch( ("cl"+_NxM+"_seedIeta").c_str(),     &_clNxM_seedIeta);
    _tree -> Branch( ("cl"+_NxM+"_seedIphi").c_str(),     &_clNxM_seedIphi);
    _tree -> Branch( ("cl"+_NxM+"_seedEta").c_str(),      &_clNxM_seedEta);
    _tree -> Branch( ("cl"+_NxM+"_seedPhi").c_str(),      &_clNxM_seedPhi);
    _tree -> Branch( ("cl"+_NxM+"_isBarrel").c_str(),     &_clNxM_isBarrel);
    _tree -> Branch( ("cl"+_NxM+"_isOverlap").c_str(),    &_clNxM_isOverlap);
    _tree -> Branch( ("cl"+_NxM+"_isEndcap").c_str(),     &_clNxM_isEndcap);
    _tree -> Branch( ("cl"+_NxM+"_tauMatchIdx").c_str(),  &_clNxM_tauMatchIdx);
    _tree -> Branch( ("cl"+_NxM+"_jetMatchIdx").c_str(),  &_clNxM_jetMatchIdx);
    _tree -> Branch( ("cl"+_NxM+"_totalEm").c_str(),      &_clNxM_totalEm);
    _tree -> Branch( ("cl"+_NxM+"_totalHad").c_str(),     &_clNxM_totalHad);
    _tree -> Branch( ("cl"+_NxM+"_totalEt").c_str(),      &_clNxM_totalEt);
    _tree -> Branch( ("cl"+_NxM+"_totalEgEt").c_str(),    &_clNxM_totalEgEt);
    _tree -> Branch( ("cl"+_NxM+"_totalIem").c_str(),     &_clNxM_totalIem);
    _tree -> Branch( ("cl"+_NxM+"_totalIhad").c_str(),    &_clNxM_totalIhad);
    _tree -> Branch( ("cl"+_NxM+"_totalIet").c_str(),     &_clNxM_totalIet);
    _tree -> Branch( ("cl"+_NxM+"_totalEgIet").c_str(),   &_clNxM_totalEgIet);
    _tree -> Branch( ("cl"+_NxM+"_towerEta").c_str(),     &_clNxM_towerEta);
    _tree -> Branch( ("cl"+_NxM+"_towerPhi").c_str(),     &_clNxM_towerPhi);
    _tree -> Branch( ("cl"+_NxM+"_towerEm").c_str(),      &_clNxM_towerEm);
    _tree -> Branch( ("cl"+_NxM+"_towerHad").c_str(),     &_clNxM_towerHad);
    _tree -> Branch( ("cl"+_NxM+"_towerEt").c_str(),      &_clNxM_towerEt);
    _tree -> Branch( ("cl"+_NxM+"_towerEgEt").c_str(),    &_clNxM_towerEgEt);
    _tree -> Branch( ("cl"+_NxM+"_towerIeta").c_str(),    &_clNxM_towerIeta);
    _tree -> Branch( ("cl"+_NxM+"_towerIphi").c_str(),    &_clNxM_towerIphi);
    _tree -> Branch( ("cl"+_NxM+"_towerIem").c_str(),     &_clNxM_towerIem);
    _tree -> Branch( ("cl"+_NxM+"_towerIhad").c_str(),    &_clNxM_towerIhad);
    _tree -> Branch( ("cl"+_NxM+"_towerIet").c_str(),     &_clNxM_towerIet);
    _tree -> Branch( ("cl"+_NxM+"_towerEgIet").c_str(),   &_clNxM_towerEgIet);
    _tree -> Branch( ("cl"+_NxM+"_towerNeg").c_str(),     &_clNxM_towerNeg);

    return;
}

void L1CaloTauNtuplizer::endJob() { return; }

void L1CaloTauNtuplizer::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) { return; }

void L1CaloTauNtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    this -> Initialize();

    _evtNumber = iEvent.id().event();
    _runNumber = iEvent.id().run();

    iEvent.getByToken(CaloClustersNxMToken, CaloClustersNxMHandle);
    iEvent.getByToken(HGClustersToken, HGClustersHandle);
    iEvent.getByToken(minatedTausToken, minatedTausHandle);
    iEvent.getByToken(squareTausToken, squareTausHandle);
    iEvent.getByToken(genTausToken, genTausHandle);
    
    const TowerHelper::TowerClustersCollection& CaloClustersNxM = *CaloClustersNxMHandle;
    const HGClusterHelper::HGClustersCollection& HGClusters = *HGClustersHandle;
    const TauHelper::TausCollection& minatedTaus = *minatedTausHandle;
    const GenHelper::GenTausCollection& genTaus = *genTausHandle;

    if (DEBUG)
    {
        std::cout << "***************************************************************************************************************************************" << std::endl;
        std::cout << " ** total number of NxM clusters = " << CaloClustersNxM.size() << std::endl;
        std::cout << " ** total number of hgc clusters = " << HGClusters.size() << std::endl;
        std::cout << " ** total number of taus = " << minatedTaus.size() << std::endl;
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

        // Perform geometrical matching of NxM CaloClusters
        int matchedCluIdx = -99;
        float dR2min = 0.25;
        for (long unsigned int cluIdx = 0; cluIdx < CaloClustersNxM.size(); cluIdx++)
        {
            TowerHelper::TowerCluster cluNxM = CaloClustersNxM[cluIdx];

            float dEta = cluNxM.seedEta - tau.visEta;
            float dPhi = reco::deltaPhi(cluNxM.seedPhi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = cluIdx;
            }

            if (DEBUG)
            {
                printf("         - NxM TOWER CLU et %i eta %f phi %f em %i had %i dEta %f dPhi %f dR2 %f (%i)\n",
                    cluNxM.totalIet,
                    cluNxM.seedEta,
                    cluNxM.seedPhi,
                    cluNxM.totalIem,
                    cluNxM.totalIhad,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TowerHelper::TowerCluster& writable_cluNxM = const_cast<TowerHelper::TowerCluster&>(CaloClustersNxM[matchedCluIdx]);
            writable_cluNxM.tauMatchIdx = tauIdx;
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

        if (DEBUG) { std::cout << "       ----------------------------------------------------------------------------------------------------------- " << std::endl; }
        
        // Perform geometrical matching of minatedl1Taus clusters
        matchedCluIdx = -99;
        dR2min = 0.25;
        for (long unsigned int minatedl1tauIdx = 0; minatedl1tauIdx < minatedTaus.size(); minatedl1tauIdx++)
        {
            TauHelper::Tau minatedl1tau = minatedTaus[minatedl1tauIdx];

            float dEta = minatedl1tau.eta - tau.visEta;
            float dPhi = reco::deltaPhi(minatedl1tau.phi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 <= dR2min)
            {
                dR2min = dR2;
                matchedCluIdx = minatedl1tauIdx;
            }

            if (DEBUG)
            {
                printf("         - HGC CLU pt %f eta %f phi %f dEta %f dPhi %f dR2 %f (%i)\n",
                    minatedl1tau.pt,
                    minatedl1tau.eta,
                    minatedl1tau.phi,
                    dEta,
                    dPhi,
                    dR2,
                    matchedCluIdx);
            }
        }
        if (matchedCluIdx != -99)
        {
            TauHelper::Tau& writable_minatedl1tau =  const_cast<TauHelper::Tau&>(minatedTaus[matchedCluIdx]);
            writable_minatedl1tau.tauMatchIdx = tauIdx;
        }

        if (DEBUG) { std::cout << "\n       *********************************************************************************************************** \n" << std::endl; }

    } // end loop on GenTaus

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

    // Fill TauMinator L1Taus
    for (long unsigned int tauIdx = 0; tauIdx < minatedTaus.size(); tauIdx++)
    {
        TauHelper::Tau tau = minatedTaus[tauIdx];

        _minatedl1tau_pt.push_back(tau.pt);
        _minatedl1tau_eta.push_back(tau.eta);
        _minatedl1tau_phi.push_back(tau.phi);
        _minatedl1tau_clusterIdx.push_back(tau.clusterIdx);
        _minatedl1tau_isBarrel.push_back(tau.isBarrel);
        _minatedl1tau_isEndcap.push_back(tau.isEndcap);
        _minatedl1tau_IDscore.push_back(tau.IDscore);
        _minatedl1tau_tauMatchIdx.push_back(tau.tauMatchIdx);
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

    // Fill NxM CaloCluster branches
    for (long unsigned int cluIdx = 0; cluIdx < CaloClustersNxM.size(); cluIdx++)
    {
        TowerHelper::TowerCluster cluNxM = CaloClustersNxM[cluIdx];

        _clNxM_calibPt.push_back(cluNxM.calibPt);
        _clNxM_IDscore.push_back(cluNxM.IDscore);
        _clNxM_barrelSeeded.push_back(cluNxM.barrelSeeded);
        _clNxM_nHits.push_back(cluNxM.nHits);
        _clNxM_seedIeta.push_back(cluNxM.seedIeta);
        _clNxM_seedIphi.push_back(cluNxM.seedIphi);
        _clNxM_seedEta.push_back(cluNxM.seedEta);
        _clNxM_seedPhi.push_back(cluNxM.seedPhi);
        _clNxM_isBarrel.push_back(cluNxM.isBarrel);
        _clNxM_isOverlap.push_back(cluNxM.isOverlap);
        _clNxM_isEndcap.push_back(cluNxM.isEndcap);
        _clNxM_tauMatchIdx.push_back(cluNxM.tauMatchIdx);
        _clNxM_jetMatchIdx.push_back(cluNxM.jetMatchIdx);
        _clNxM_totalEm.push_back(cluNxM.totalEm);
        _clNxM_totalHad.push_back(cluNxM.totalHad);
        _clNxM_totalEt.push_back(cluNxM.totalEt);
        _clNxM_totalEgEt.push_back(cluNxM.totalEgEt);
        _clNxM_totalIem.push_back(cluNxM.totalIem);
        _clNxM_totalIhad.push_back(cluNxM.totalIhad);
        _clNxM_totalIet.push_back(cluNxM.totalIet);
        _clNxM_totalEgIet.push_back(cluNxM.totalEgIet);
        
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

        for (long unsigned int i = 0; i < cluNxM.towerHits.size(); ++i)
        {
            tmp_towerEta.push_back(cluNxM.towerHits[i].towerEta);
            tmp_towerPhi.push_back(cluNxM.towerHits[i].towerPhi);
            tmp_towerEm.push_back(cluNxM.towerHits[i].towerEm);
            tmp_towerHad.push_back(cluNxM.towerHits[i].towerHad);
            tmp_towerEt.push_back(cluNxM.towerHits[i].towerEt);
            tmp_towerEgEt.push_back(cluNxM.towerHits[i].l1egTowerEt);
            tmp_towerIeta.push_back(cluNxM.towerHits[i].towerIeta);
            tmp_towerIphi.push_back(cluNxM.towerHits[i].towerIphi);
            tmp_towerIem.push_back(cluNxM.towerHits[i].towerIem);
            tmp_towerIhad.push_back(cluNxM.towerHits[i].towerIhad);
            tmp_towerIet.push_back(cluNxM.towerHits[i].towerIet);
            tmp_towerEgIet.push_back(cluNxM.towerHits[i].l1egTowerIet);
            tmp_towerNeg.push_back(cluNxM.towerHits[i].nL1eg);

            nEGs += cluNxM.towerHits[i].nL1eg;
        }

        _clNxM_towerEta.push_back(tmp_towerEta);
        _clNxM_towerPhi.push_back(tmp_towerPhi);
        _clNxM_towerEm.push_back(tmp_towerEm);
        _clNxM_towerHad.push_back(tmp_towerHad);
        _clNxM_towerEt.push_back(tmp_towerEt);
        _clNxM_towerEgEt.push_back(tmp_towerEgEt);
        _clNxM_towerIeta.push_back(tmp_towerIeta);
        _clNxM_towerIphi.push_back(tmp_towerIphi);
        _clNxM_towerIem.push_back(tmp_towerIem);
        _clNxM_towerIhad.push_back(tmp_towerIhad);
        _clNxM_towerIet.push_back(tmp_towerIet);
        _clNxM_towerEgIet.push_back(tmp_towerEgIet);
        _clNxM_towerNeg.push_back(tmp_towerNeg);

        _clNxM_nEGs.push_back(nEGs);
    }

    // Fill HGCluster branches
    for (long unsigned int hgcluIdx = 0; hgcluIdx < HGClusters.size(); hgcluIdx++)
    {
        HGClusterHelper::HGCluster hgclu = HGClusters[hgcluIdx];

        _cl3d_calibPt.push_back(hgclu.calibPt);
        _cl3d_IDscore.push_back(hgclu.IDscore);
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

DEFINE_FWK_MODULE(L1CaloTauNtuplizer);