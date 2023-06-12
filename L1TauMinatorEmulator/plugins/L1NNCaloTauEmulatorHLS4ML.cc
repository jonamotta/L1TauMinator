/* -*- C++ -*-

Package: L1CaloTrigger
Class: l1tNNCaloTauEmulatorHLS4ML
Frinedly name: The TauMinator

\class l1tNNCaloTauEmulatorHLS4ML l1tNNCaloTauEmulatorHLS4ML.cc

Description: 
Perform firmware-exact emulation of the l1tNNCaloTauProducer
that implements the NN Calo Tau.
(Perform reconstruction and identification of tau 
candidates at L1 Trigger with a CNN.)

Implementation:
The implementation is done forseeing the integration
of the algorithm in the GCT Sum card. This means that
the full detector information can be accessed at the same
time (ECAL, HCAL, HGCAL full eta-phi coverage). This will
come in the form of arrays of towers and clusters.
Given that the emulators of the upstream algortihms are
not fully determined yet, this emulator takes as input
the simulation-based information, manipulates it software-like
to pruduce the arrays of towers and clusters as they should
be available in the GCT sum card. Only then the actual
emulation of the algorithm arrives.

Original Author: Jona Motta
Created: Tue June 7th 2023

*/

#include <iostream>
#include <vector>
#include <cmath>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "ap_int.h"
#include "ap_fixed.h"
#include "../../../hls4mlEmulatorExtras/include/hls4ml/emulator.h"
// #include "hls4ml/emulator.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterEgID.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class l1tNNCaloTauEmulatorHLS4ML : public edm::stream::EDProducer<> {
public:
  explicit l1tNNCaloTauEmulatorHLS4ML(const edm::ParameterSet&);

private:
  // ----fixed LSBs, Nbits, scales, and types----
  static constexpr int INTPHI_PI = 36;
  static constexpr int INTPHI_2PI = 2 * INTPHI_PI;
  static constexpr float IETAPHI_LSB = M_PI / INTPHI_PI;
  
  static constexpr float SHAPEFEAT_LSB = 0.0000153; // pow(2, -16)
  static constexpr float SZZ_LSB = SHAPEFEAT_LSB * 100;
  static constexpr float ETAPHI_LSB = M_PI / 720;
  static constexpr float ETAHGCAL_OFFSET = 1.321; // inferred from hgcal schematics
  static constexpr float IETAHGCAL_LSB = 0.0845; // inferred from simulation
  static constexpr float PUID_LSB = 0.00390625; // pow(2, -8)
  static constexpr float MEANZ_OFFSET = 321.05; // inferred from hgcal schematics
  static constexpr int IETAHGCAL_OFFSET = 17;
  static constexpr float MEANZ_LSB = 0.5;
  static constexpr float PTET_LSB = 0.25;
  static constexpr float CM2MM = 10;
  static constexpr int R2cone = 0.25 / IETAPHI_LSB / IETAPHI_LSB;

  static constexpr int SHAPEFEAT_W = 16; // maximum forseen per shape
  static constexpr int DETAPHI_W = 11;
  static constexpr int DIETAPHI_W = 8;
  static constexpr int IETAPHI_W = 7;
  static constexpr int SHOWLEN_W = 6;
  static constexpr int ETAPHI_W = 11; // precision going to correlator
  static constexpr int MEANZ_W = 12;
  static constexpr int PUID_W = 9;
  
  static constexpr int PT_W = 14;
  static constexpr int PT_I = 12;
  static constexpr int ET_W = 10;
  static constexpr int ET_I = 8;
  
  typedef ap_ufixed<PT_W, PT_I, AP_TRN, AP_SAT> Pt_t;
  typedef ap_fixed<ET_W, ET_I,AP_TRN, AP_SAT> Et_t;

  typedef ap_uint<SHAPEFEAT_W> ShapeFeat_t; // careful not all shapes might be unsigned
  typedef ap_int<DIETAPHI_W> dIEtaPhi_t;
  typedef ap_int<DIETAPHI_W> dEtaPhi_t;
  typedef ap_uint<SHOWLEN_W> ShowLen_t;
  typedef ap_int<ETAPHI_W> EtaPhi_t;
  typedef ap_uint<IETAPHI_W> IPhi_t;
  typedef ap_int<IETAPHI_W> IEta_t;
  typedef ap_uint<MEANZ_W> Meanz_t;
  typedef ap_int<PUID_W> PUid_t; // this means that id>-0.10 corresponds to id>25
  
  // ----fixed dimensions of tower clusters----
  const int seedIdx = 22;
  const int IEta_dim = 5;
  const int IPhi_dim = 9;
  const int Eta_limit = 33;

  // ----fixed dimensions of hls4ml models input----
  const int ETA_IMG_EXT = 5;
  const int PHI_IMG_EXT = 9;
  const int NMBR_FILTER = 3;
  const int SEED_DIMENS = 2;
  const int NMBR_SHAPES = 8;
  const int CNNOUT_DIM = 26;

  //----edm control---
  void produce(edm::Event&, const edm::EventSetup&) override;

  //----private functions----
  dIEtaPhi_t tower_dIPhi(IPhi_t iPhi_1, IPhi_t iPhi_2);
  dIEtaPhi_t tower_dIEta(IEta_t iEta_1, IEta_t iEta_2);
  dIEtaPhi_t tw2cl_dIPhi(EtaPhi_t iPhi_1, IPhi_t iPhi_2);
  dIEtaPhi_t tw2cl_dIEta(EtaPhi_t iEta_1, IEta_t iEta_2);
  dEtaPhi_t tw2cl_dPhi(EtaPhi_t iPhi_1, IPhi_t iPhi_2);
  dEtaPhi_t tw2cl_dEta(EtaPhi_t iEta_1, IEta_t iEta_2);
  IEta_t makeEndcapHwIEta(float eta);
  IPhi_t makeEndcapHwIPhi(float phi);
  float apfixedQuantizer(float inputF, float LSB, int nbits);
  int apintQuantizer(float inputF, float LSB, int nbits);
  float inputScaler(float inputF, std::string feature);
  float correctInputEtaCl3d(float eta);
  float correctInputMeanzCl3d(float meanz);

  inline float floatPt(Pt_t pt) { return pt.to_float(); }
  inline float floatEt(Et_t et) { return et.to_float(); }
  inline float floatEta(EtaPhi_t eta) { return eta.to_float() * ETAPHI_LSB; }
  inline float floatPhi(EtaPhi_t phi) { return phi.to_float() * ETAPHI_LSB; }
  inline float floatShape(ShapeFeat_t shape) { return shape.to_float() * SHAPEFEAT_LSB; };
  inline float floatSzz(ShapeFeat_t szz) { return szz.to_float() * SZZ_LSB; };
  inline float floatMeanZ(Meanz_t meanz) { return meanz.to_float() * MEANZ_LSB + MEANZ_OFFSET; };
  inline float floatMeanZHgcalCoord(Meanz_t meanz) { return meanz.to_float() * MEANZ_LSB; };
  inline float floatPuId(PUid_t pu) { return pu.to_float() * PUID_LSB; };
  float floatIEta(IEta_t eta);
  float floatIPhi(IPhi_t phi);

  template <int W>
  ap_int<W> ap_abs(ap_int<W> x);
  template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
  ap_ufixed<W,I> ap_abs(ap_fixed<W, I, _AP_Q, _AP_O> x);

  //----tokens and handles----
  edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowersToken;
  edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

  edm::EDGetToken hgcalTowersToken;
  edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;

  edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> HGClusterToken;
  edm::Handle<l1t::HGCalMulticlusterBxCollection> HGClusterHandle;

  //----private variables----
  enum class UseEmInterp { No, EmOnly, AllKeepHad, AllKeepTot };
  UseEmInterp scenario;
  StringCutObjectSelector<l1t::HGCalMulticluster> preEmId;
  l1tpf::HGC3DClusterEgID VsPuId;

  double EcalEtMinForClustering;
  double HcalEtMinForClustering;
  double EtMinForSeeding;
  double EtaRestriction;
  double PuidThr;

  std::string CNNmodel_CB_path;
  std::string DNNident_CB_path;
  std::string DNNcalib_CB_path;

  std::string CNNmodel_CE_path;
  std::string DNNident_CE_path;
  std::string DNNcalib_CE_path;
  std::string FeatScaler_CE_path;
  boost::property_tree::ptree FeatScaler_CE;

  tensorflow::GraphDef* CNNmodel_CB;
  tensorflow::GraphDef* DNNident_CB;
  tensorflow::GraphDef* DNNcalib_CB;

  tensorflow::Session* CNNmodel_CBsession;
  tensorflow::Session* DNNident_CBsession;
  tensorflow::Session* DNNcalib_CBsession;

  tensorflow::GraphDef* CNNmodel_CE;
  tensorflow::GraphDef* DNNident_CE;
  tensorflow::GraphDef* DNNcalib_CE;

  tensorflow::Session* CNNmodel_CEsession;
  tensorflow::Session* DNNident_CEsession;
  tensorflow::Session* DNNcalib_CEsession;

  hls4mlEmulator::ModelLoader CNNmodel_CB_loader;
  hls4mlEmulator::ModelLoader DNNident_CB_loader;
  hls4mlEmulator::ModelLoader DNNcalib_CB_loader;

  std::shared_ptr<hls4mlEmulator::Model> CNNmodel_CB_model;
  std::shared_ptr<hls4mlEmulator::Model> DNNident_CB_model;
  std::shared_ptr<hls4mlEmulator::Model> DNNcalib_CB_model;

  hls4mlEmulator::ModelLoader CNNmodel_CE_loader;
  hls4mlEmulator::ModelLoader DNNident_CE_loader;
  hls4mlEmulator::ModelLoader DNNcalib_CE_loader;

  std::shared_ptr<hls4mlEmulator::Model> CNNmodel_CE_model;
  std::shared_ptr<hls4mlEmulator::Model> DNNident_CE_model;
  std::shared_ptr<hls4mlEmulator::Model> DNNcalib_CE_model;

  double IdWp90_CB;
  double IdWp95_CB;
  double IdWp99_CB;

  double IdWp90_CE;
  double IdWp95_CE;
  double IdWp99_CE;

  PUid_t intPuidThr;
  IEta_t intEtaRestriction;

  // Class for the towers info as they should be in GCT
  class SimpleTowerHit {
  public:
    IEta_t towerIeta = 0;
    IPhi_t towerIphi = 0;
    Et_t towerEm = 0.;
    Et_t towerHad = 0.;
    Et_t l1egTowerEt = 0.;
    Et_t towerEt = 0.;
    ap_uint<1> isBarrel = 0x1;
    ap_uint<1> stale = 0x0;
    ap_uint<1> stale4seed = 0x0;
  };

  // Class for the clusters info as they should arrive from HGCAL
  class SimpleHGCluster {
  public:
    Pt_t pt;
    EtaPhi_t eta;
    EtaPhi_t phi;
    ShowLen_t showerlength;
    ShowLen_t coreshowerlength;
    ShapeFeat_t spptot;
    ShapeFeat_t szz;
    ShapeFeat_t srrtot;
    Meanz_t meanz;
    PUid_t PUid;
    ap_uint<1> stale = 0x0;
  };

  // Classes for the tower clusters
  class SimplifiedTower {
  public:
    Et_t towerEm = 0.;
    Et_t towerHad = 0.;
    Et_t l1egTowerEt = 0.;
    IEta_t ieta = 0; // REMOVE - DEBUG ONLY
    IPhi_t iphi = 0; // REMOVE - DEBUG ONLY

    void fill(SimpleTowerHit Tower) {
      towerEm = Tower.towerEm;
      towerHad = Tower.towerHad;
      l1egTowerEt = Tower.l1egTowerEt;
      ieta = Tower.towerIeta; // REMOVE - DEBUG ONLY
      iphi = Tower.towerIphi; // REMOVE - DEBUG ONLY
    }
  };

  class InputTowerCluster {
  public:
    SimplifiedTower towerHits[45];
    ap_uint<1> barrelSeeded = 0x0;
    ap_uint<1> filled[45];

    void fill(int idx, SimpleTowerHit Tower) {
      towerHits[idx].fill(Tower);
      filled[idx] = 0x1;
    }

    void init() {
      SimplifiedTower emptyT;
      std::fill(towerHits, towerHits+44, emptyT);
      std::fill(filled, filled+44, 0x0);
    }

  };

  class InputTowerCluster_pstn {
  public:
    IEta_t seedIeta = 0;
    IPhi_t seedIphi = 0;

    void fill(SimpleTowerHit Tower) {
      seedIeta = Tower.towerIeta;
      seedIphi = Tower.towerIphi;
    }
  };

  // FIXME : now variables are in GCT precision, they should be in NN precision
  // after scaling, i.e. something like ap_fixed<16, 6, AP_TRN, AP_SAT>
  class InputHGCluster {
  public:
    Pt_t pt;
    EtaPhi_t eta;
    EtaPhi_t phi; // REMOVE - DEBUG ONLY
    ShowLen_t showerlength;
    ShowLen_t coreshowerlength;
    ShapeFeat_t spptot;
    ShapeFeat_t szz;
    ShapeFeat_t srrtot;
    Meanz_t meanz;

    void fill(SimpleHGCluster Cluster) {
      pt = Cluster.pt;
      eta = Cluster.eta;
      phi = Cluster.phi; // REMOVE - DEBUG ONLY
      showerlength = Cluster.showerlength;
      coreshowerlength = Cluster.coreshowerlength;
      spptot = Cluster.spptot;
      szz = Cluster.szz;
      srrtot = Cluster.srrtot;
      meanz = Cluster.meanz;
    }
  };

};

/*
████████ ██   ██ ██████     ████████  █████  ██   ██ ███    ███ ██ ███    ██  █████  ████████  ██████  ██████  
   ██    ██   ██ ██            ██    ██   ██ ██   ██ ████  ████ ██ ████   ██ ██   ██    ██    ██    ██ ██   ██ 
   ██    ███████ █████         ██    ███████ ██   ██ ██ ████ ██ ██ ██ ██  ██ ███████    ██    ██    ██ ██████  
   ██    ██   ██ ██            ██    ██   ██ ██   ██ ██  ██  ██ ██ ██  ██ ██ ██   ██    ██    ██    ██ ██   ██ 
   ██    ██   ██ ██████        ██    ██   ██ ███████ ██      ██ ██ ██   ████ ██   ██    ██     ██████  ██    ██
*/

// ----Constructor and Destructor -----
l1tNNCaloTauEmulatorHLS4ML::l1tNNCaloTauEmulatorHLS4ML(const edm::ParameterSet& iConfig)
    : l1TowersToken(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
      hgcalTowersToken(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("hgcalTowers"))),

      HGClusterToken(
          consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalClusters"))),
      scenario(UseEmInterp::No),
      preEmId(iConfig.getParameter<std::string>("preEmId")),
      VsPuId(iConfig.getParameter<edm::ParameterSet>("VsPuId")),

      EcalEtMinForClustering(iConfig.getParameter<double>("EcalEtMinForClustering")),
      HcalEtMinForClustering(iConfig.getParameter<double>("HcalEtMinForClustering")),
      EtMinForSeeding(iConfig.getParameter<double>("EtMinForSeeding")),
      EtaRestriction(iConfig.getParameter<double>("EtaRestriction")),
      PuidThr(iConfig.getParameter<double>("PuidThr")),

      CNNmodel_CB_path(iConfig.getParameter<std::string>("CNNmodel_CB_path")),
      DNNident_CB_path(iConfig.getParameter<std::string>("DNNident_CB_path")),
      DNNcalib_CB_path(iConfig.getParameter<std::string>("DNNcalib_CB_path")),
      CNNmodel_CE_path(iConfig.getParameter<std::string>("CNNmodel_CE_path")),
      DNNident_CE_path(iConfig.getParameter<std::string>("DNNident_CE_path")),
      DNNcalib_CE_path(iConfig.getParameter<std::string>("DNNcalib_CE_path")),
      FeatScaler_CE_path(iConfig.getParameter<std::string>("FeatScaler_CE_path")),

      CNNmodel_CB_loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<std::string>("CNNmodel_CB_loader"))),
      DNNident_CB_loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<std::string>("DNNident_CB_loader"))),
      DNNcalib_CB_loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<std::string>("DNNcalib_CB_loader"))),
      CNNmodel_CE_loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<std::string>("CNNmodel_CE_loader"))),
      DNNident_CE_loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<std::string>("DNNident_CE_loader"))),
      DNNcalib_CE_loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<std::string>("DNNcalib_CE_loader"))),

      IdWp90_CB(iConfig.getParameter<double>("IdWp90_CB")),
      IdWp95_CB(iConfig.getParameter<double>("IdWp95_CB")),
      IdWp99_CB(iConfig.getParameter<double>("IdWp99_CB")),

      IdWp90_CE(iConfig.getParameter<double>("IdWp90_CE")),
      IdWp95_CE(iConfig.getParameter<double>("IdWp95_CE")),
      IdWp99_CE(iConfig.getParameter<double>("IdWp99_CE")) {
  // Create sessions for Tensorflow inferece
  CNNmodel_CB = tensorflow::loadGraphDef(edm::FileInPath(CNNmodel_CB_path).fullPath());
  CNNmodel_CBsession = tensorflow::createSession(CNNmodel_CB);

  DNNident_CB = tensorflow::loadGraphDef(edm::FileInPath(DNNident_CB_path).fullPath());
  DNNident_CBsession = tensorflow::createSession(DNNident_CB);

  DNNcalib_CB = tensorflow::loadGraphDef(edm::FileInPath(DNNcalib_CB_path).fullPath());
  DNNcalib_CBsession = tensorflow::createSession(DNNcalib_CB);

  CNNmodel_CE = tensorflow::loadGraphDef(edm::FileInPath(CNNmodel_CE_path).fullPath());
  CNNmodel_CEsession = tensorflow::createSession(CNNmodel_CE);

  DNNident_CE = tensorflow::loadGraphDef(edm::FileInPath(DNNident_CE_path).fullPath());
  DNNident_CEsession = tensorflow::createSession(DNNident_CE);

  DNNcalib_CE = tensorflow::loadGraphDef(edm::FileInPath(DNNcalib_CE_path).fullPath());
  DNNcalib_CEsession = tensorflow::createSession(DNNcalib_CE);

  // Load models for hls4mlExternals ineference
  CNNmodel_CB_model = CNNmodel_CB_loader.load_model()
  DNNident_CB_model = DNNident_CB_loader.load_model()
  DNNcalib_CB_model = DNNcalib_CB_loader.load_model()
  CNNmodel_CE_model = CNNmodel_CE_loader.load_model()
  DNNident_CE_model = DNNident_CE_loader.load_model()
  DNNcalib_CE_model = DNNcalib_CE_loader.load_model()

  // Read features scaler
  boost::property_tree::read_json(edm::FileInPath(FeatScaler_CE_path).fullPath(), FeatScaler_CE);

  // Initialize HGCAL BDTs
  if (!VsPuId.method().empty()) {
    VsPuId.prepareTMVA();
  }

  intPuidThr = apintQuantizer(PuidThr, PUID_LSB, PUID_W);
  intEtaRestriction = apintQuantizer(EtaRestriction, IETAPHI_LSB, IETAPHI_W);

  // Create produced outputs
  produces<BXVector<l1t::Tau>>("L1NNCaloTauCollectionBXV");

  // Settings output
  std::cout << "EtaRestriction = " << EtaRestriction << " (" << intEtaRestriction << ") , EtMinForSeeding = " << EtMinForSeeding
            << " , HcalTpEtMin = " << HcalEtMinForClustering << " , EcalTpEtMin = " << EcalEtMinForClustering
            << " , PuidThr = " << PuidThr << "(" << intPuidThr << ")"
            << std::endl;
}

void l1tNNCaloTauEmulatorHLS4ML::produce(edm::Event& iEvent, const edm::EventSetup& eSetup) {
  // Output collection
  std::unique_ptr<BXVector<l1t::Tau>> L1NNCaloTauCollectionBXV(new l1t::TauBxCollection);

  // Create and Fill collection of all calotowers and their attributes
  std::vector<SimpleTowerHit> l1CaloTowers;

  iEvent.getByToken(l1TowersToken, l1CaloTowerHandle);
  int warnings = 0;
  for (auto& hit : *l1CaloTowerHandle.product()) {
    // Skip this weird towers and store warning
    if (hit.towerIEta() == -1016 && hit.towerIPhi() == -962) {
      warnings += 1;
      continue;
    }

    SimpleTowerHit l1Hit;
    l1Hit.isBarrel = 0x1;
    l1Hit.l1egTowerEt = apfixedQuantizer(hit.l1egTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerEm = apfixedQuantizer(hit.ecalTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerHad = apfixedQuantizer(hit.hcalTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerEt = apfixedQuantizer(hit.ecalTowerEt() + hit.hcalTowerEt() + hit.l1egTowerEt(), PTET_LSB, ET_W);
    l1Hit.towerIeta = hit.towerIEta();
    l1Hit.towerIphi = hit.towerIPhi();

    l1CaloTowers.push_back(l1Hit);
  }
  if (warnings != 0) {
    std::cout << " ** WARNING : FOUND " << warnings << " TOWERS WITH towerIeta=-1016 AND towerIphi=-962" << std::endl;
  }

  iEvent.getByToken(hgcalTowersToken, hgcalTowersHandle);
  for (auto& hit : *hgcalTowersHandle.product()) {
    SimpleTowerHit l1Hit;
    l1Hit.isBarrel = 0x0;
    l1Hit.l1egTowerEt = 0.0;
    l1Hit.towerEm = apfixedQuantizer(hit.etEm(), PTET_LSB, ET_W) ;
    l1Hit.towerHad = apfixedQuantizer(hit.etHad(), PTET_LSB, ET_W) ;
    l1Hit.towerEt = apfixedQuantizer(hit.etEm() + hit.etHad(), PTET_LSB, ET_W) ;
    l1Hit.towerIeta = makeEndcapHwIEta(hit.eta());
    l1Hit.towerIphi = makeEndcapHwIPhi(hit.phi());

    l1CaloTowers.push_back(l1Hit);
  } 

  // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
  std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const SimpleTowerHit& a, SimpleTowerHit& b) {
    return a.towerEt > b.towerEt;
  });

  // Create and Fill the collection of 3D clusters and their attributes
  std::vector<SimpleHGCluster> AllHGClusters;
  iEvent.getByToken(HGClusterToken, HGClusterHandle);

  for (auto cl3dIt = HGClusterHandle->begin(0); cl3dIt != HGClusterHandle->end(0); ++cl3dIt) {
    auto& cl3d = *cl3dIt;

    // Implement cl3d PU ID as done in
    // https://github.com/cms-sw/cmssw/blob/master/L1Trigger/Phase2L1ParticleFlow/plugins/PFClusterProducerFromHGC3DClusters.cc#L120
    bool isEM = preEmId(*cl3dIt);
    l1t::PFCluster cluster(cl3d.pt(), cl3d.eta(), cl3d.phi(), cl3d.hOverE());
    if (scenario == UseEmInterp::EmOnly)  // for emID objs, use EM interp as pT and set H = 0
    {
      if (isEM) {
        float pt_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
        float hoe_new = 0.;
        cluster = l1t::PFCluster(pt_new, cl3d.eta(), cl3d.phi(), hoe_new, isEM);
      }
    } else if (scenario == UseEmInterp::AllKeepHad)  // for all objs, replace EM part with EM interp, preserve H
    {
      float had_old = cl3d.pt() - cluster.emEt();
      float em_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
      float pt_new = had_old + em_new;
      float hoe_new = em_new > 0 ? (had_old / em_new) : -1;
      cluster = l1t::PFCluster(pt_new, cl3d.eta(), cl3d.phi(), hoe_new, isEM);
    } else if (scenario == UseEmInterp::AllKeepTot)  // for all objs, replace EM part with EM interp, preserve pT
    {
      float em_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
      float hoe_new = em_new > 0 ? (cl3d.pt() / em_new - 1) : -1;
      cluster = l1t::PFCluster(cl3d.pt(), cl3d.eta(), cl3d.phi(), hoe_new, isEM);
    }

    float idScore = -1.;
    if (!VsPuId.method().empty()) {
      int id = VsPuId.passID(*cl3dIt, cluster);
      idScore = cluster.egVsPUMVAOut();
    }

    float eta_hgcalCoord = correctInputEtaCl3d(cl3d.eta());
    float meanz_hgcalCoord = correctInputMeanzCl3d(cl3d.zBarycenter());

    SimpleHGCluster HGCluster;
    HGCluster.pt = apfixedQuantizer(cl3d.pt(), PTET_LSB, PT_W);
    HGCluster.eta = apintQuantizer(eta_hgcalCoord, ETAPHI_LSB, ETAPHI_W);
    HGCluster.phi = apintQuantizer(cl3d.phi(), ETAPHI_LSB, ETAPHI_W);
    HGCluster.showerlength = cl3d.showerLength();
    HGCluster.coreshowerlength = cl3d.coreShowerLength();
    HGCluster.spptot = apintQuantizer(cl3d.sigmaPhiPhiTot(), SHAPEFEAT_LSB, SHAPEFEAT_W);
    HGCluster.szz = apintQuantizer(cl3d.sigmaZZ(), SZZ_LSB, SHAPEFEAT_W);
    HGCluster.srrtot = apintQuantizer(cl3d.sigmaRRTot(), SHAPEFEAT_LSB, SHAPEFEAT_W);
    HGCluster.meanz = apintQuantizer(meanz_hgcalCoord, MEANZ_LSB, MEANZ_W);
    HGCluster.PUid = apintQuantizer(idScore, PUID_LSB, PUID_W);

    AllHGClusters.push_back(HGCluster);
  }

  // Order the collection in pt (the input to the GCT will be pt ordered)
  std::sort(begin(AllHGClusters), end(AllHGClusters), [](const SimpleHGCluster& a, SimpleHGCluster& b) {
    return a.pt > b.pt;
  });

  /*
  // END OF SOFTWARE-LIKE SECTION
  // up to here treated inputs from simulation with SW precision
  // to massage them into the HW precision varibales as they are
  // forseen (roughly) to be available at the GCT Sum card level
  // ------------------------------------------------------------- */

  // Make NxM TowerClusters and HGClusters collections for TauMinator
  std::vector<InputTowerCluster> l1TowerClustersNxM_CB;
  std::vector<InputTowerCluster_pstn> l1TowerClustersNxM_CB_pstn;
  std::vector<InputTowerCluster> l1TowerClustersNxM_CE;
  std::vector<InputTowerCluster_pstn> l1TowerClustersNxM_CE_pstn;
  std::vector<InputHGCluster> HGClusters;

  // Supporting collection of endcap clusters before cl3d matching
  std::vector<InputTowerCluster> AllL1TowerClustersNxM_CE;
  std::vector<InputTowerCluster_pstn> AllL1TowerClustersNxM_CE_pstn;

  int Nclusters_CB = 0;
  int AllNclusters_CE = 0;
  bool caloTauSeedingFinished = false;
  // Loop for seeding of clNxM objects
  while (!caloTauSeedingFinished) {
    InputTowerCluster clNxM; clNxM.init();
    InputTowerCluster_pstn clNxM_pstn;
    bool seeded = false;

    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip seeding in towers that would make the cluster extend in HF
      // Skip l1CaloTowers which are already used by this clusters' mask
      if (ap_abs(l1CaloTower.towerIeta) > Eta_limit || l1CaloTower.stale4seed) {
        continue;
      }

      // If not seded do the seeding
      if (!seeded) {
        // The leading unused tower has ET < min, stop jet clustering
        if (l1CaloTower.towerEt < EtMinForSeeding) {
          caloTauSeedingFinished = true;
          continue;
        }

        clNxM.fill(seedIdx, l1CaloTower);
        clNxM_pstn.fill(l1CaloTower);
        if (l1CaloTower.isBarrel) {
          clNxM.barrelSeeded = 0x1;
        }

        l1CaloTower.stale4seed = 0x1;
        l1CaloTower.stale = 0x1;
        seeded = true;

        continue;
      }

      dIEtaPhi_t d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM_pstn.seedIeta);
      dIEtaPhi_t d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM_pstn.seedIphi);

      // Stale tower for seeding if it would lead to overalp between clusters
      if ((ap_abs(d_iEta) <= IEta_dim - 1 && ap_abs(d_iPhi) <= IPhi_dim - 1)) {
        l1CaloTower.stale4seed = 0x1;
      }

    }  // End for loop over TPs

    // Pushback seeds split in barrel and endcap
    if (seeded) {
      if (clNxM.barrelSeeded) {
        l1TowerClustersNxM_CB.push_back(clNxM);
        l1TowerClustersNxM_CB_pstn.push_back(clNxM_pstn);
        Nclusters_CB++;
      } else {
        AllL1TowerClustersNxM_CE.push_back(clNxM);
        AllL1TowerClustersNxM_CE_pstn.push_back(clNxM_pstn);
        AllNclusters_CE++;
      }
    }

  }  // End while loop of TowerClusters seeding


  // Loop for barrel NxM TowerClusters clustering starting from the seeds
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CB; clNxMIdx++) {

    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip l1CaloTowers which are already used
      if (l1CaloTower.stale) {
        continue;
      }

      dIEtaPhi_t d_iEta = tower_dIEta(l1CaloTower.towerIeta, l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIeta);
      dIEtaPhi_t d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIphi);
      int hitIdx = d_iEta * 9 + d_iPhi + seedIdx;

      // Cluster all towers in a NxM towers mask
      if ((ap_abs(d_iEta) <= (IEta_dim - 1) / 2 && ap_abs(d_iPhi) <= (IPhi_dim - 1) / 2)) {
        
        l1TowerClustersNxM_CB[clNxMIdx].fill(hitIdx, l1CaloTower);
        l1CaloTower.stale = 0x1;
      }

    }  // End for loop over TPs

  }  // End while loop of barrel TowerClusters creation



/*  // cluster numpy array printout for nice displays
  for (int idxi = 0; idxi < Nclusters_CB; idxi++)
  {
    InputTowerCluster clu5x9 = l1TowerClustersNxM_CB[idxi];
    std::cout << "cluster_"<<idxi<<"_f = np.array([";
    for (long unsigned int j = 0; j < 45; ++j) { std::cout<<"["<<clu5x9.towerHits[j].ieta.to_float()<<","<<clu5x9.towerHits[j].iphi.to_float()<<"],"; }
    std::cout << "]\n" << std::endl;
  }

  for (int idxi = 0; idxi < Nclusters_CB; idxi++)
  {
    InputTowerCluster clu5x9 = l1TowerClustersNxM_CB[idxi];
    InputTowerCluster_pstn clu5x9_pstn = l1TowerClustersNxM_CB_pstn[idxi];
    std::cout << "cluster_"<<idxi<<"_seed = np.array(["<<clu5x9.towerHits[22].ieta.to_float()<<","<<clu5x9.towerHits[22].iphi.to_float()<<"])\n"<< std::endl;
    std::cout << "cluster_"<<idxi<<"_pstn = np.array(["<<clu5x9_pstn.seedIeta.to_float()<<","<<clu5x9_pstn.seedIphi.to_float()<<"])\n"<< std::endl;
  }
*/


  // In the endcap cross-loop over clNxM and cl3d to match them
  // (we can do it before full clustering just using the seed info)
  int Nclusters_CE = 0;
  for (int clNxMIdx = 0; clNxMIdx < AllNclusters_CE; clNxMIdx++) {

    bool matched = false;
    for (auto& HGCluster : AllHGClusters) {
      // In case the clNxM or HGCluster have already been matched just continue through the list to the end
      // only use cl3ds above 4GeV and above -0.10 pu id
      if (matched || HGCluster.stale || HGCluster.pt < Pt_t(4.) || HGCluster.PUid < intPuidThr) {
        continue;
      }

      dIEtaPhi_t d_iEta = tw2cl_dIEta(HGCluster.eta, AllL1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta);
      dIEtaPhi_t d_iPhi = tw2cl_dIPhi(HGCluster.phi, AllL1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi);
      matched = d_iEta*d_iEta + d_iPhi*d_iPhi < R2cone;

      if (matched) {
        HGCluster.stale = 0x1;
        InputHGCluster cl3d; cl3d.fill(HGCluster);
        HGClusters.push_back(cl3d);
        l1TowerClustersNxM_CE.push_back(AllL1TowerClustersNxM_CE[clNxMIdx]);
        l1TowerClustersNxM_CE_pstn.push_back(AllL1TowerClustersNxM_CE_pstn[clNxMIdx]);
        Nclusters_CE++;
      }

    }  // End for loop over cl3ds

  }  // End for loop over clNxM


  // Loop for endcap matched NxM TowerClusters clustering starting from the seeds just found
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CE; clNxMIdx++) {

    for (auto& l1CaloTower : l1CaloTowers) {
      // Skip l1CaloTowers which are already used
      if (l1CaloTower.stale) {
        continue;
      }

      dIEtaPhi_t d_iEta = tower_dIEta(l1CaloTower.towerIeta, l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta);
      dIEtaPhi_t d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi);
      int hitIdx = d_iEta * 9 + d_iPhi + seedIdx;

      // Cluster all towers in a NxM towers mask
      if ((ap_abs(d_iEta) <= (IEta_dim - 1) / 2 && ap_abs(d_iPhi) <= (IPhi_dim - 1) / 2)) {
        
        l1TowerClustersNxM_CE[clNxMIdx].fill(hitIdx, l1CaloTower);
        l1CaloTower.stale = 0x1;
      }

    }  // End for loop over TPs

  }  // End while loop of barrel TowerClusters creation


/*  // cluster numpy array printout for nice displays
  for (int idxi = 0; idxi < Nclusters_CE; idxi++)
  {
    InputTowerCluster clu5x9 = l1TowerClustersNxM_CE[idxi];
    std::cout << "cluster_"<<idxi<<"_f = np.array([";
    for (long unsigned int j = 0; j < 45; ++j) { std::cout<<"["<<clu5x9.towerHits[j].ieta<<","<<clu5x9.towerHits[j].iphi<<"],"; }
    std::cout << "]\n" << std::endl;
  }

  for (int idxi = 0; idxi < Nclusters_CE; idxi++)
  {
    InputHGCluster cl3d = HGClusters[idxi]; 

    ap_int<8> coarseiPhi = cl3d.phi / (IETAPHI_LSB / ETAPHI_LSB); 
    if (coarseiPhi <= 0) { coarseiPhi += INTPHI_2PI; }
    // add 1 because tower 0 does not exist
    else { coarseiPhi += 1; }
  
    EtaPhi_t iEta_1 = cl3d.eta;
    EtaPhi_t etaHgcalOffset = apintQuantizer(ETAHGCAL_OFFSET, ETAPHI_LSB, ETAPHI_W);
    EtaPhi_t lastBarrelTowerEdge = apintQuantizer(IETAHGCAL_OFFSET*IETAPHI_LSB, ETAPHI_LSB, ETAPHI_W);
    // correct for the diffence in towers_edge and hgcal_edge
    // after this the clusters' eta is w.r.t. the last barrel tower edge
    if (iEta_1 > 0) { iEta_1 -= lastBarrelTowerEdge - etaHgcalOffset; }
    else { iEta_1 += lastBarrelTowerEdge - etaHgcalOffset; }
    IEta_t coarseiEta = iEta_1 / (IETAHGCAL_LSB / ETAPHI_LSB);
    // add 1 to correct for truncation error
    if (coarseiEta < 0) { coarseiEta -= IETAHGCAL_OFFSET + 1; }
    else { coarseiEta += IETAHGCAL_OFFSET + 1; }

    InputTowerCluster_pstn clu5x9_pstn = l1TowerClustersNxM_CE_pstn[idxi];
    std::cout << "cluster_"<<idxi<<"_mtch = np.array(["<<coarseiEta<<","<<coarseiPhi<<"])\n"<< std::endl;
    std::cout << "cluster_"<<idxi<<"_pstn = np.array(["<<clu5x9_pstn.seedIeta<<","<<clu5x9_pstn.seedIphi<<"])\n"<< std::endl;
  }
*/


  // Barrel TauMinator application
  tensorflow::setLogging("2");
  int batchSize_CB = (int)(Nclusters_CB);
  tensorflow::TensorShape imageShape_CB({batchSize_CB, IEta_dim, IPhi_dim, 3});
  tensorflow::TensorShape positionShape_CB({batchSize_CB, 2});
  tensorflow::Tensor TowerClusterImage_CB(tensorflow::DT_FLOAT, imageShape_CB);
  tensorflow::Tensor TowerClusterPosition_CB(tensorflow::DT_FLOAT, positionShape_CB);

  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CB; clNxMIdx++) {
    // Fill inputs for Tensorflow inference
    for (int eta = 0; eta < IEta_dim; ++eta) {
      for (int phi = 0; phi < IPhi_dim; ++phi) {
        int towerIdx = eta * IPhi_dim + phi;
        TowerClusterImage_CB.tensor<float, 4>()(clNxMIdx, eta, phi, 0) = l1TowerClustersNxM_CB[clNxMIdx].towerHits[towerIdx].l1egTowerEt.to_float();
        TowerClusterImage_CB.tensor<float, 4>()(clNxMIdx, eta, phi, 1) = l1TowerClustersNxM_CB[clNxMIdx].towerHits[towerIdx].towerEm.to_float();
        TowerClusterImage_CB.tensor<float, 4>()(clNxMIdx, eta, phi, 2) = l1TowerClustersNxM_CB[clNxMIdx].towerHits[towerIdx].towerHad.to_float();
      }
    }

    TowerClusterPosition_CB.tensor<float, 2>()(clNxMIdx, 0) = floatIEta(l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIeta);
    TowerClusterPosition_CB.tensor<float, 2>()(clNxMIdx, 1) = floatIPhi(l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIphi);
  }

  // Apply CNN model
  tensorflow::NamedTensorList CNNmodel_CBinputList = {{"TowerClusterImage", TowerClusterImage_CB},
                                                      {"TowerClusterPosition", TowerClusterPosition_CB}};
  std::vector<tensorflow::Tensor> CNNmodel_CBoutputs;
  tensorflow::run(
      CNNmodel_CBsession, CNNmodel_CBinputList, {"TauMinator_CB_conv/middleMan/concat"}, &CNNmodel_CBoutputs);
  tensorflow::NamedTensorList DNN_CBinputsList = {{"middleMan", CNNmodel_CBoutputs[0]}};

  // Apply DNN for identification
  std::vector<tensorflow::Tensor> DNN_CBoutputsIdent;
  tensorflow::run(
      DNNident_CBsession, DNN_CBinputsList, {"TauMinator_CB_ident/sigmoid_IDout/Sigmoid"}, &DNN_CBoutputsIdent);

  // Apply DNN for calibration
  std::vector<tensorflow::Tensor> DNN_CBoutputsCalib;
  tensorflow::run(DNNcalib_CBsession, DNN_CBinputsList, {"TauMinator_CB_calib/DNNout/MatMul"}, &DNN_CBoutputsCalib);

  // Fill the output collection of L1 taus
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CB; clNxMIdx++) {
    int seedIeta = l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIeta;
    int seedIphi = l1TowerClustersNxM_CB_pstn[clNxMIdx].seedIphi;

    if (seedIeta > intEtaRestriction) {
      continue;
    }

    float tau_IDscore = DNN_CBoutputsIdent[0].matrix<float>()(0, clNxMIdx);
    float tau_calibPt = DNN_CBoutputsCalib[0].matrix<float>()(0, clNxMIdx);
    float tau_eta = floatIEta(seedIeta);
    float tau_phi = floatIPhi(seedIphi);

    // Assign increasing quality to higher scoring candidates
    int quality = 0;
    // 99% WP
    if (tau_IDscore > IdWp99_CB) {
      quality = 1;
    }
    // 95% WP
    if (tau_IDscore > IdWp95_CB) {
      quality = 2;
    }
    // 90% WP
    if (tau_IDscore > IdWp90_CB) {
      quality = 3;
    }

    reco::Candidate::PolarLorentzVector tauP4 =
        reco::Candidate::PolarLorentzVector(tau_calibPt, tau_eta, tau_phi, 0);

    // store ID score multiplied by 10E4 to have good precision even using the Phase1 tau int iso format
    // (this is stored just in case for possible additional offline studies)
    // tau initialisation =  (p4,    pt,          eta,     phi,     qual,    iso)
    l1t::Tau l1Tau = l1t::Tau(tauP4, tau_calibPt, tau_eta, tau_phi, quality, tau_IDscore * 10E4);
    l1Tau.setTowerIEta(seedIeta);
    l1Tau.setTowerIPhi(seedIphi);
    // l1Tau.setRawEt(clNxM.rawEt); // FIXME : not really needed tbh

    L1NNCaloTauCollectionBXV->push_back(0, l1Tau);
  }


  // Endcap TauMinator application
  int batchSize_CE = (int)(Nclusters_CE);
  tensorflow::TensorShape imageShape_CE({batchSize_CE, IEta_dim, IPhi_dim, 3});
  tensorflow::TensorShape positionShape_CE({batchSize_CE, 2});
  tensorflow::TensorShape cl3dfeatShape_CE({batchSize_CE, 8});
  tensorflow::Tensor TowerClusterImage_CE(tensorflow::DT_FLOAT, imageShape_CE);
  tensorflow::Tensor TowerClusterPosition_CE(tensorflow::DT_FLOAT, positionShape_CE);
  tensorflow::Tensor Cl3dShapeFeatures_CE(tensorflow::DT_FLOAT, cl3dfeatShape_CE);

  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CE; clNxMIdx++) {
    // Indexing of cl3ds is the same as the one of clNxMs
    InputHGCluster HGClu = HGClusters[clNxMIdx];

    // Fill inputs for Tensorflow inference
    for (int eta = 0; eta < IEta_dim; ++eta) {
      for (int phi = 0; phi < IPhi_dim; ++phi) {
        int towerIdx = eta * IPhi_dim + phi;
        TowerClusterImage_CE.tensor<float, 4>()(clNxMIdx, eta, phi, 0) = l1TowerClustersNxM_CE[clNxMIdx].towerHits[towerIdx].l1egTowerEt.to_float();
        TowerClusterImage_CE.tensor<float, 4>()(clNxMIdx, eta, phi, 1) = l1TowerClustersNxM_CE[clNxMIdx].towerHits[towerIdx].towerEm.to_float();
        TowerClusterImage_CE.tensor<float, 4>()(clNxMIdx, eta, phi, 2) = l1TowerClustersNxM_CE[clNxMIdx].towerHits[towerIdx].towerHad.to_float();
      }
    }

    TowerClusterPosition_CE.tensor<float, 2>()(clNxMIdx, 0) = floatIEta(l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta);
    TowerClusterPosition_CE.tensor<float, 2>()(clNxMIdx, 1) = floatIPhi(l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi);

    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 0) = inputScaler(HGClu.pt.to_float(), "pt");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 1) = inputScaler(abs(floatEta(HGClu.eta)), "eta");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 2) = inputScaler(HGClu.showerlength.to_float(), "showerlength");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 3) = inputScaler(HGClu.coreshowerlength.to_float(), "coreshowerlength");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 4) = inputScaler(floatShape(HGClu.spptot), "spptot");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 5) = inputScaler(floatSzz(HGClu.szz), "szz");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 6) = inputScaler(floatShape(HGClu.srrtot), "srrtot");
    Cl3dShapeFeatures_CE.tensor<float, 2>()(clNxMIdx, 7) = inputScaler(floatMeanZHgcalCoord(HGClu.meanz), "meanz");

    // std::cout << "  cluster " << std::endl;
    // std::cout << "        pt " << HGClu.pt.to_float() << " - scaled pt " << inputScaler(HGClu.pt.to_float(), "pt") << std::endl;
    // std::cout << "        eta " << abs(floatEta(HGClu.eta)) << " - scaled eta " << inputScaler(abs(floatEta(HGClu.eta)), "eta") << std::endl;
    // std::cout << "        sl " << HGClu.showerlength << " - scaled sl " << inputScaler(HGClu.showerlength.to_float(), "showerlength") << std::endl;
    // std::cout << "        csl " << HGClu.coreshowerlength << " - scaled csl " << inputScaler(HGClu.coreshowerlength.to_float(), "coreshowerlength") << std::endl;
    // std::cout << "        spp " << floatShape(HGClu.spptot) << " - scaled spp " << inputScaler(floatShape(HGClu.spptot), "spptot") << std::endl;
    // std::cout << "        szz " << floatSzz(HGClu.szz) << " - scaled szz " << inputScaler(floatSzz(HGClu.szz), "szz") << std::endl;
    // std::cout << "        srr " << floatShape(HGClu.srrtot) << " - scaled srr " << inputScaler(floatShape(HGClu.srrtot), "srrtot") << std::endl;
    // std::cout << "        meanz " << floatMeanZHgcalCoord(HGClu.meanz) << " - scaled meanz " << inputScaler(floatMeanZHgcalCoord(HGClu.meanz), "meanz") << std::endl;

  }

  // Apply CNN model
  tensorflow::NamedTensorList CNNmodel_CEinputList = {{"TowerClusterImage", TowerClusterImage_CE},
                                                      {"TowerClusterPosition", TowerClusterPosition_CE},
                                                      {"AssociatedCl3dFeatures", Cl3dShapeFeatures_CE}};
  std::vector<tensorflow::Tensor> CNNmodel_CEoutputs;
  tensorflow::run(
      CNNmodel_CEsession, CNNmodel_CEinputList, {"TauMinator_CE_conv/middleMan/concat"}, &CNNmodel_CEoutputs);
  tensorflow::NamedTensorList DNN_CEinputsList = {{"middleMan", CNNmodel_CEoutputs[0]}};

  // Apply DNN for identification
  std::vector<tensorflow::Tensor> DNN_CEoutputsIdent;
  tensorflow::run(
      DNNident_CEsession, DNN_CEinputsList, {"TauMinator_CE_ident/sigmoid_IDout/Sigmoid"}, &DNN_CEoutputsIdent);

  // Apply DNN for calibration
  std::vector<tensorflow::Tensor> DNN_CEoutputsCalib;
  tensorflow::run(DNNcalib_CEsession, DNN_CEinputsList, {"TauMinator_CE_calib/DNNout/MatMul"}, &DNN_CEoutputsCalib);



  ap_ufixed<10,8> TowerClusterImage_CB_hls4ml[ETA_IMG_EXT*PHI_IMG_EXT*NMBR_FILTER];
  ap_fixed<9,3> TowerClusterPosition_CB_hls4ml[SEED_DIMENS];
  ap_fixed<11,6> MiddleMan_CB_hls4ml[CNNOUT_DIM];
  ap_fixed<10,9> OutputCalibration_CB_hls4ml[1];
  ap_fixed<8,1,AP_RND,AP_SAT> OutputIdentification_CB_hls4ml[1];

  CNNmodel_CB_model->prepare_input(TowerClusterImage_CB_hls4ml, TowerClusterPosition_CB_hls4ml);
  CNNmodel_CB_model->predict();
  CNNmodel_CB_model->read_result(MiddleMan_CB_hls4ml);

  DNNident_CB_model->prepare_input(MiddleMan_CB_hls4ml);
  DNNident_CB_model->predict();
  DNNident_CB_model->read_result(OutputIdentification_CB_hls4ml);

  DNNcalib_CB_model->prepare_input(MiddleMan_CB_hls4ml);
  DNNcalib_CB_model->predict();
  DNNcalib_CB_model->read_result(OutputCalibration_CB_hls4ml);



  ap_ufixed<10,8> TowerClusterImage_CE_hls4ml[ETA_IMG_EXT*PHI_IMG_EXT*NMBR_FILTER];
  ap_fixed<9,3> TowerClusterPosition_CE_hls4ml[SEED_DIMENS];
  ap_fixed<16,6> Cl3dShapeFeatures_CE_hls4ml[NMBR_SHAPES];
  ap_fixed<11,6> MiddleMan_CE_hls4ml[CNNOUT_DIM+NMBR_SHAPES];
  ap_fixed<10,9> OutputCalibration_CE_hls4ml[1];
  ap_fixed<8,1,AP_RND,AP_SAT> OutputIdentification_CE_hls4ml[1];

  CNNmodel_CE_model->prepare_input(TowerClusterImage_CE_hls4ml, TowerClusterPosition_CE_hls4ml, Cl3dShapeFeatures_CE_hls4ml);
  CNNmodel_CE_model->predict();
  CNNmodel_CE_model->read_result(MiddleMan_CE_hls4ml);

  DNNident_CE_model->prepare_input(MiddleMan_CE_hls4ml);
  DNNident_CE_model->predict();
  DNNident_CE_model->read_result(OutputIdentification_CE_hls4ml);

  DNNcalib_CE_model->prepare_input(MiddleMan_CE_hls4ml);
  DNNcalib_CE_model->predict();
  DNNcalib_CE_model->read_result(OutputCalibration_CE_hls4ml);



  // Fill the output collection of L1 taus
  for (int clNxMIdx = 0; clNxMIdx < Nclusters_CE; clNxMIdx++) {
    int seedIeta = l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIeta;
    int seedIphi = l1TowerClustersNxM_CE_pstn[clNxMIdx].seedIphi;

    if (seedIeta > intEtaRestriction) {
      continue;
    }

    float tau_IDscore = DNN_CEoutputsIdent[0].matrix<float>()(0, clNxMIdx);
    float tau_calibPt = DNN_CEoutputsCalib[0].matrix<float>()(0, clNxMIdx);
    float tau_eta = floatIEta(seedIeta);
    float tau_phi = floatIPhi(seedIphi);

    // Assign increasing quality to higher scoring candidates
    int quality = 0;
    // 99% WP
    if (tau_IDscore > IdWp99_CE) {
      quality = 1;
    }
    // 95% WP
    if (tau_IDscore > IdWp95_CE) {
      quality = 2;
    }
    // 90% WP
    if (tau_IDscore > IdWp90_CE) {
      quality = 3;
    }

    reco::Candidate::PolarLorentzVector tauP4 =
        reco::Candidate::PolarLorentzVector(tau_calibPt, tau_eta, tau_phi, 0);

    // store ID score multiplied by 10E4 to have good precision even using the Phase1 tau int iso format
    // (this is stored just in case for possible additional offline studies)
    // tau initialisation =  (p4,    pt,          eta,     phi,     qual,    iso)
    l1t::Tau l1Tau = l1t::Tau(tauP4, tau_calibPt, tau_eta, tau_phi, quality, tau_IDscore * 10E4);
    l1Tau.setTowerIEta(seedIeta);
    l1Tau.setTowerIPhi(seedIphi);
    // l1Tau.setRawEt(clNxM.rawEt); // FIXME : not really needed tbh

    L1NNCaloTauCollectionBXV->push_back(0, l1Tau);
  }

  // Fill output
  iEvent.put(std::move(L1NNCaloTauCollectionBXV), "L1NNCaloTauCollectionBXV");

}  // End of produce function

l1tNNCaloTauEmulatorHLS4ML::dIEtaPhi_t l1tNNCaloTauEmulatorHLS4ML::tower_dIPhi(IPhi_t iPhi_1, IPhi_t iPhi_2) {
  dIEtaPhi_t result = iPhi_1 - iPhi_2;
  if (result > INTPHI_PI) {
    result -= INTPHI_2PI;
  }
  if (result <= -INTPHI_PI) {
    result += INTPHI_2PI;
  }

  return result;
}

l1tNNCaloTauEmulatorHLS4ML::dIEtaPhi_t l1tNNCaloTauEmulatorHLS4ML::tower_dIEta(IEta_t iEta_1, IEta_t iEta_2) {
  ap_int<12> mult = iEta_1 * iEta_2;
  dIEtaPhi_t result = iEta_1 - iEta_2;
  if (mult < 0) {
    if (iEta_1 > 0) {
      result -= 1;
    }
    else {
      result += 1;
    }
  }

  return result;
}

l1tNNCaloTauEmulatorHLS4ML::dIEtaPhi_t l1tNNCaloTauEmulatorHLS4ML::tw2cl_dIPhi(EtaPhi_t iPhi_1, IPhi_t iPhi_2) {
  ap_int<8> coarseiPhi_1 = iPhi_1 / (IETAPHI_LSB / ETAPHI_LSB); 
  if (coarseiPhi_1 <= 0) {
    coarseiPhi_1 += INTPHI_2PI;
  }
  // add 1 because tower 0 does not exist
  else {
    coarseiPhi_1 += 1;
  }
  
  return tower_dIPhi(coarseiPhi_1, iPhi_2);
}

l1tNNCaloTauEmulatorHLS4ML::dIEtaPhi_t l1tNNCaloTauEmulatorHLS4ML::tw2cl_dIEta(EtaPhi_t iEta_1, IEta_t iEta_2) {
  EtaPhi_t etaHgcalOffset = apintQuantizer(ETAHGCAL_OFFSET, ETAPHI_LSB, ETAPHI_W);
  EtaPhi_t lastBarrelTowerEdge = apintQuantizer(IETAHGCAL_OFFSET*IETAPHI_LSB, ETAPHI_LSB, ETAPHI_W);

  // correct for the diffence in towers_edge and hgcal_edge
  // after this the clusters' eta is w.r.t. the last barrel tower edge
  if (iEta_1 > 0) {
    iEta_1 -= lastBarrelTowerEdge - etaHgcalOffset;
  }
  else {
    iEta_1 += lastBarrelTowerEdge - etaHgcalOffset;
  }

  IEta_t coarseiEta_1 = iEta_1 / (IETAHGCAL_LSB / ETAPHI_LSB);

  // add 1 to correct for truncation error
  if (coarseiEta_1 < 0) {
    coarseiEta_1 -= IETAHGCAL_OFFSET + 1;
  }
  else {
    coarseiEta_1 += IETAHGCAL_OFFSET + 1;
  }

  return tower_dIEta(coarseiEta_1, iEta_2);
}

l1tNNCaloTauEmulatorHLS4ML::IEta_t l1tNNCaloTauEmulatorHLS4ML::makeEndcapHwIEta(float eta) {
  IEta_t ieta = floor(eta/IETAHGCAL_LSB);

  if (ieta < 0) {
    ieta += 1; // FIXME : +1 NEEDED?
  }

  return ieta;
}

l1tNNCaloTauEmulatorHLS4ML::IPhi_t l1tNNCaloTauEmulatorHLS4ML::makeEndcapHwIPhi(float phi) {
  if (phi < 0) {
    phi += 2 * M_PI;
  }

  // shift up by half an LSB
  phi += IETAPHI_LSB/2; // FIXME : +-1 NEEDED?

  return floor(phi/IETAPHI_LSB);
}

template <int W>
ap_int<W> l1tNNCaloTauEmulatorHLS4ML::ap_abs(ap_int<W> x) {
  ap_int<W> result;
  if (x < 0) {
    result = -x;
  } else {
    result = x;
  }
  return result;
}

template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
ap_ufixed<W,I> l1tNNCaloTauEmulatorHLS4ML::ap_abs(ap_fixed<W, I, _AP_Q, _AP_O> x) {
  ap_ufixed<W,I> result;
  if (x < 0) {
    result = -x;
  } else {
    result = x;
  }
  return result;
}

float l1tNNCaloTauEmulatorHLS4ML::apfixedQuantizer(float inputF, float LSB, int nbits) {
  return min(floor(inputF / LSB), float(pow(2, nbits) - 1)) * LSB;
}

int l1tNNCaloTauEmulatorHLS4ML::apintQuantizer(float inputF, float LSB, int nbits) {
  return min(floor(inputF / LSB), float(pow(2, nbits) - 1));
}

float l1tNNCaloTauEmulatorHLS4ML::inputScaler(float inputF, std::string feature) {
  float mean = FeatScaler_CE.get_child(feature).get<float>("mean");
  float std = FeatScaler_CE.get_child(feature).get<float>("std");

  return (inputF - mean) / std;
}

float l1tNNCaloTauEmulatorHLS4ML::correctInputEtaCl3d(float eta) {
  if (eta > 0) {
    eta -= ETAHGCAL_OFFSET;
  }
  else {
    eta += ETAHGCAL_OFFSET;
  }

  return eta;
}

float l1tNNCaloTauEmulatorHLS4ML::correctInputMeanzCl3d(float meanz) {
  return CM2MM * (abs(meanz) - MEANZ_OFFSET);
}

float l1tNNCaloTauEmulatorHLS4ML::floatIEta(IEta_t eta) {
  // transform eta of towers from integer to float, correcting for different barrel/endcap LSB
  float feta;
  if (abs(eta) > IETAHGCAL_OFFSET) {
    if (eta>0) {
      feta = IETAHGCAL_OFFSET * IETAPHI_LSB + (eta.to_float() - IETAHGCAL_OFFSET) * IETAHGCAL_LSB;
    }
    else {
      feta = -IETAHGCAL_OFFSET * IETAPHI_LSB + (eta.to_float() + IETAHGCAL_OFFSET) * IETAHGCAL_LSB;
    }
  }
  else {
    feta = eta.to_float() * IETAPHI_LSB;
  }

  // shift by half a tower to consider the tower center instead of the edge
  if (feta > 0) {
    feta -= IETAPHI_LSB / 2;
  }
  else {
    feta += IETAPHI_LSB / 2;
  }

  return feta;  // FIXME : +-1 NEEDED?

}

float l1tNNCaloTauEmulatorHLS4ML::floatIPhi(IPhi_t phi) {
  float fphi = phi.to_float();
  // add 2pi + 1 because tower 0 does not exist
  if (fphi > INTPHI_PI) {
    fphi -= INTPHI_2PI + 1; // FIXME : +1 NEEDED?
  }

  fphi *= IETAPHI_LSB;

  // shift by half a tower to consider the tower center instead of the edge
  if (fphi > 0) {
    fphi -= IETAPHI_LSB / 2;
  }
  else {
    fphi += IETAPHI_LSB / 2;
  }

  return fphi; // FIXME : +-1 NEEDED?
}


DEFINE_FWK_MODULE(l1tNNCaloTauEmulatorHLS4ML);
