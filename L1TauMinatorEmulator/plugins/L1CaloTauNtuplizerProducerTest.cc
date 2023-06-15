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

#include "L1TauMinator/DataFormats/interface/GenHelper.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


// class L1CaloTauNtuplizerProducerTest : public edm::stream::EDAnalyzer<> {
class L1CaloTauNtuplizerProducerTest : public edm::EDAnalyzer {
    public:
        explicit L1CaloTauNtuplizerProducerTest(const edm::ParameterSet&);
        virtual ~L1CaloTauNtuplizerProducerTest();

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
        edm::EDGetTokenT<l1t::TauBxCollection> TauMinatorTausToken;
        edm::Handle<BXVector<l1t::Tau>> TauMinatorTausHandle;

        edm::EDGetTokenT<GenHelper::GenTausCollection> genTausToken;
        edm::Handle<GenHelper::GenTausCollection> genTausHandle;

        edm::EDGetTokenT<GenHelper::GenTausCollection> genTausModToken;
        edm::Handle<GenHelper::GenTausCollection> genTausModHandle;

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

        std::vector<float> _tau_mod_eta;
        std::vector<float> _tau_mod_phi;
        std::vector<float> _tau_mod_pt;
        std::vector<float> _tau_mod_e;
        std::vector<float> _tau_mod_m;
        std::vector<float> _tau_mod_visEta;
        std::vector<float> _tau_mod_visPhi;
        std::vector<float> _tau_mod_visPt;
        std::vector<float> _tau_mod_sumEta;
        std::vector<float> _tau_mod_sumPhi;
        std::vector<float> _tau_mod_sumPt;
        std::vector<float> _tau_mod_visE;
        std::vector<float> _tau_mod_visM;

        std::vector<float>  _minatedl1tau_pt;
        std::vector<float>  _minatedl1tau_eta;
        std::vector<float>  _minatedl1tau_phi;
        std::vector<float>  _minatedl1tau_quality;
        std::vector<float>  _minatedl1tau_IDscore;
        std::vector<float>  _minatedl1tau_towerIEta;
        std::vector<float>  _minatedl1tau_towerIPhi;
        std::vector<float>  _minatedl1tau_rawEt;

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
L1CaloTauNtuplizerProducerTest::L1CaloTauNtuplizerProducerTest(const edm::ParameterSet& iConfig)
    : TauMinatorTausToken(consumes<BXVector<l1t::Tau>>(iConfig.getParameter<edm::InputTag>("TauMinatorTaus"))),
      genTausToken(consumes<GenHelper::GenTausCollection>(iConfig.getParameter<edm::InputTag>("genTaus"))),
      genTausModToken(consumes<GenHelper::GenTausCollection>(iConfig.getParameter<edm::InputTag>("genTausMod"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{
    _treeName = iConfig.getParameter<std::string>("treeName");
    this -> Initialize();
    return;
}

L1CaloTauNtuplizerProducerTest::~L1CaloTauNtuplizerProducerTest() {}

void L1CaloTauNtuplizerProducerTest::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void L1CaloTauNtuplizerProducerTest::Initialize()
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

    _tau_mod_eta.clear();
    _tau_mod_phi.clear();
    _tau_mod_pt.clear();
    _tau_mod_e.clear();
    _tau_mod_m.clear();
    _tau_mod_visEta.clear();
    _tau_mod_visPhi.clear();
    _tau_mod_visPt.clear();
    _tau_mod_sumEta.clear();
    _tau_mod_sumPhi.clear();
    _tau_mod_sumPt.clear();
    _tau_mod_visE.clear();
    _tau_mod_visM.clear();

    _minatedl1tau_pt.clear();
    _minatedl1tau_eta.clear();
    _minatedl1tau_phi.clear();
    _minatedl1tau_quality.clear();
    _minatedl1tau_IDscore.clear();
    _minatedl1tau_towerIEta.clear();
    _minatedl1tau_towerIPhi.clear();
    _minatedl1tau_rawEt.clear();
}

void L1CaloTauNtuplizerProducerTest::beginJob()
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

    _tree -> Branch("tau_mod_eta",      &_tau_mod_eta);
    _tree -> Branch("tau_mod_phi",      &_tau_mod_phi);
    _tree -> Branch("tau_mod_pt",       &_tau_mod_pt);
    _tree -> Branch("tau_mod_e",        &_tau_mod_e);
    _tree -> Branch("tau_mod_m",        &_tau_mod_m);
    _tree -> Branch("tau_mod_visEta",   &_tau_mod_visEta);
    _tree -> Branch("tau_mod_visPhi",   &_tau_mod_visPhi);
    _tree -> Branch("tau_mod_visPt",    &_tau_mod_visPt);
    _tree -> Branch("tau_mod_sumEta",   &_tau_mod_sumEta);
    _tree -> Branch("tau_mod_sumPhi",   &_tau_mod_sumPhi);
    _tree -> Branch("tau_mod_sumPt",    &_tau_mod_sumPt);
    _tree -> Branch("tau_mod_visE",     &_tau_mod_visE);
    _tree -> Branch("tau_mod_visM",     &_tau_mod_visM);

    _tree -> Branch("minatedl1tau_pt",         &_minatedl1tau_pt);
    _tree -> Branch("minatedl1tau_eta",        &_minatedl1tau_eta);
    _tree -> Branch("minatedl1tau_phi",        &_minatedl1tau_phi);
    _tree -> Branch("minatedl1tau_quality",    &_minatedl1tau_quality);
    _tree -> Branch("minatedl1tau_IDscore",    &_minatedl1tau_IDscore);
    _tree -> Branch("minatedl1tau_towerIEta",  &_minatedl1tau_towerIEta);
    _tree -> Branch("minatedl1tau_towerIPhi",  &_minatedl1tau_towerIPhi);
    _tree -> Branch("minatedl1tau_rawEt",      &_minatedl1tau_rawEt);

    return;
}

void L1CaloTauNtuplizerProducerTest::endJob()
{ 
    if (brokenTensorflowPrediction)
    {
        std::cout << "** ERROR: the Tensorflow inference went ballistic in this job! Please re-run it!" << std::endl;
    }

    return;
}

void L1CaloTauNtuplizerProducerTest::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) { return; }

void L1CaloTauNtuplizerProducerTest::analyze(const edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    this -> Initialize();

    _evtNumber = iEvent.id().event();
    _runNumber = iEvent.id().run();

    iEvent.getByToken(TauMinatorTausToken, TauMinatorTausHandle);
    iEvent.getByToken(genTausToken, genTausHandle);
    iEvent.getByToken(genTausModToken, genTausModHandle);
    
    const GenHelper::GenTausCollection& genTaus = *genTausHandle;
    const GenHelper::GenTausCollection& genTaus_mod = *genTausModHandle;

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

    for (long unsigned int tauIdx = 0; tauIdx < genTaus_mod.size(); tauIdx++)
    {
        GenHelper::GenTau tau = genTaus_mod[tauIdx];

        _tau_mod_eta.push_back(tau.eta);
        _tau_mod_phi.push_back(tau.phi);
        _tau_mod_pt.push_back(tau.pt);
        _tau_mod_e.push_back(tau.e);
        _tau_mod_m.push_back(tau.m);
        _tau_mod_visEta.push_back(tau.visEta);
        _tau_mod_visPhi.push_back(tau.visPhi);
        _tau_mod_visPt.push_back(tau.visPt);
        _tau_mod_sumEta.push_back(tau.sumEta);
        _tau_mod_sumPhi.push_back(tau.sumPhi);
        _tau_mod_sumPt.push_back(tau.sumPt);
        _tau_mod_visE.push_back(tau.visE);
        _tau_mod_visM.push_back(tau.visM);
    }

    // Fill TauMinator L1Taus
    for (l1t::TauBxCollection::const_iterator bx0TauIt = TauMinatorTausHandle->begin(0); bx0TauIt != TauMinatorTausHandle->end(0) ; bx0TauIt++)
    {
        const l1t::Tau& tau = *bx0TauIt;

        _minatedl1tau_pt.push_back(tau.pt());
        _minatedl1tau_eta.push_back(tau.eta());
        _minatedl1tau_phi.push_back(tau.phi());
        _minatedl1tau_quality.push_back(tau.hwQual());
        _minatedl1tau_IDscore.push_back(tau.hwIso());
        _minatedl1tau_towerIEta.push_back(tau.towerIEta());
        _minatedl1tau_towerIPhi.push_back(tau.towerIPhi());
        _minatedl1tau_rawEt.push_back(tau.rawEt());

        if (tau.pt() > 10000) { brokenTensorflowPrediction = true; }
    }

    // Fill tree
    _tree -> Fill();
}

DEFINE_FWK_MODULE(L1CaloTauNtuplizerProducerTest);