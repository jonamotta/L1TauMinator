#include <iostream>
#include <vector>
#include <cmath>

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
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
#include "L1TauMinator/DataFormats/interface/TauHelper.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <xgboost/c_api.h>


class L1CaloTauProducer : public edm::stream::EDProducer<> {
    public:
        explicit L1CaloTauProducer(const edm::ParameterSet&);

    private:
        //----edm control---
        void produce(edm::Event&, const edm::EventSetup&) override;

        //----private functions----
        int tower_dIPhi(int &iPhi_1, int &iPhi_2) const;
        int tower_dIEta(int &iEta_1, int &iEta_2) const;
        int endcap_iphi(float &phi) const;
        int endcap_ieta(float &eta) const;
        std::vector<TowerHelper::TowerHit> sortPicLike(std::vector<TowerHelper::TowerHit>) const;

        //----tokens and handles----
        edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowersToken;
        edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

        edm::EDGetToken hgcalTowersToken;
        edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;

        edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalDigisToken;
        edm::Handle<HcalTrigPrimDigiCollection> hcalDigisHandle;
        edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderTag;

        edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> HGClusterToken;
        edm::Handle<l1t::HGCalMulticlusterBxCollection> HGClusterHandle;

        //----private variables----
        int etaClusterDimension;
        int phiClusterDimension;
        int CNNfilters;
        double EcalEtMinForClustering;
        double HcalEtMinForClustering;
        double EtMinForSeeding;

        std::string CNNmodel_path;
        std::string DNNident_path;
        std::string DNNcalib_path;

        std::string XGBident_path;
        std::string XGBcalib_path;
        std::vector<std::string> XGBident_feats;
        std::vector<std::string> XGBcalib_feats;
        std::vector<double> C1calib_params;
        std::vector<double> C3calib_params;

        bool DEBUG;

        tensorflow::GraphDef* CNNmodel;
        tensorflow::GraphDef* DNNident;
        tensorflow::GraphDef* DNNcalib;

        tensorflow::Session* CNNsession;
        tensorflow::Session* DNNsessionIdent;
        tensorflow::Session* DNNsessionCalib;
        
        BoosterHandle XGBident;
        BoosterHandle XGBcalib;
};


/*
██ ███    ███ ██████  ██      ███████ ███    ███ ███████ ███    ██ ████████  █████  ████████ ██  ██████  ███    ██
██ ████  ████ ██   ██ ██      ██      ████  ████ ██      ████   ██    ██    ██   ██    ██    ██ ██    ██ ████   ██
██ ██ ████ ██ ██████  ██      █████   ██ ████ ██ █████   ██ ██  ██    ██    ███████    ██    ██ ██    ██ ██ ██  ██
██ ██  ██  ██ ██      ██      ██      ██  ██  ██ ██      ██  ██ ██    ██    ██   ██    ██    ██ ██    ██ ██  ██ ██
██ ██      ██ ██      ███████ ███████ ██      ██ ███████ ██   ████    ██    ██   ██    ██    ██  ██████  ██   ████
*/

// ----Constructor and Destructor -----
L1CaloTauProducer::L1CaloTauProducer(const edm::ParameterSet& iConfig) 
    : l1TowersToken(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
      hgcalTowersToken(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("hgcalTowers"))),
      hcalDigisToken(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigis"))),
      decoderTag(esConsumes<CaloTPGTranscoder, CaloTPGRecord>(edm::ESInputTag("", ""))),
      HGClusterToken(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalClusters"))),
      etaClusterDimension(iConfig.getParameter<int>("etaClusterDimension")),
      phiClusterDimension(iConfig.getParameter<int>("phiClusterDimension")),
      CNNfilters(iConfig.getParameter<int>("CNNfilters")),
      EcalEtMinForClustering(iConfig.getParameter<double>("EcalEtMinForClustering")),
      HcalEtMinForClustering(iConfig.getParameter<double>("HcalEtMinForClustering")),
      EtMinForSeeding(iConfig.getParameter<double>("EtMinForSeeding")),
      CNNmodel_path(iConfig.getParameter<std::string>("CNNmodel_path")),
      DNNident_path(iConfig.getParameter<std::string>("DNNident_path")),
      DNNcalib_path(iConfig.getParameter<std::string>("DNNcalib_path")),
      XGBident_path(iConfig.getParameter<std::string>("XGBident_path")),
      XGBcalib_path(iConfig.getParameter<std::string>("XGBcalib_path")),
      XGBident_feats(iConfig.getParameter<std::vector<std::string>>("XGBident_feats")),
      XGBcalib_feats(iConfig.getParameter<std::vector<std::string>>("XGBcalib_feats")),
      C1calib_params(iConfig.getParameter<std::vector<double>>("C1calib_params")),
      C3calib_params(iConfig.getParameter<std::vector<double>>("C3calib_params")),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    

    // Create sessions for Tensorflow inferece
    CNNmodel = tensorflow::loadGraphDef(CNNmodel_path);
    CNNsession = tensorflow::createSession(CNNmodel);

    DNNident = tensorflow::loadGraphDef(DNNident_path);
    DNNsessionIdent = tensorflow::createSession(DNNident);

    DNNcalib = tensorflow::loadGraphDef(DNNcalib_path);
    DNNsessionCalib = tensorflow::createSession(DNNcalib);

    // Load models for XGBoost inference
    XGBoosterCreate(NULL,0,&XGBident);
    XGBoosterLoadModel(XGBident,XGBident_path.c_str());

    XGBoosterCreate(NULL,0,&XGBcalib);
    XGBoosterLoadModel(XGBcalib,XGBcalib_path.c_str());

    // Create produced outputs
    produces<TowerHelper::TowerClustersCollection>("l1TowerClustersNxM");
    produces<HGClusterHelper::HGClustersCollection>("HGClustersCollection");
    produces<TauHelper::TausCollection>("TausCollection");

    if (DEBUG) { std::cout << "EtMinForSeeding = " << EtMinForSeeding << " , HcalTpEtMin = " << HcalEtMinForClustering << " , EcalTpEtMin = " << EcalEtMinForClustering << std::endl; }
}

void L1CaloTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill collection of all calotowers and their attributes
    std::vector<TowerHelper::TowerHit> l1CaloTowers;

    iEvent.getByToken(l1TowersToken, l1CaloTowerHandle);
    for (auto &hit : *l1CaloTowerHandle.product())
    {
        TowerHelper::TowerHit l1Hit;
        l1Hit.isBarrel     = true;
        l1Hit.l1egTowerEt  = hit.l1egTowerEt();
        l1Hit.l1egTowerIet = floor( l1Hit.l1egTowerEt/0.5 );
        l1Hit.nL1eg        = hit.nL1eg();
        l1Hit.towerEta     = hit.towerEta();
        l1Hit.towerPhi     = hit.towerPhi();
        l1Hit.towerEm      = hit.ecalTowerEt();
        l1Hit.towerHad     = hit.hcalTowerEt();
        l1Hit.towerEt      = l1Hit.towerEm + l1Hit.towerHad + l1Hit.l1egTowerEt;
        l1Hit.towerIeta    = hit.towerIEta();
        l1Hit.towerIphi    = hit.towerIPhi();
        l1Hit.towerIem     = floor( l1Hit.towerEm/0.5 );
        l1Hit.towerIhad    = floor( l1Hit.towerHad/0.5 );
        l1Hit.towerIet     = floor( (l1Hit.towerEm + l1Hit.towerHad + l1Hit.l1egTowerEt)/0.5 );

        l1CaloTowers.push_back(l1Hit);
    }

    int maxIetaHGCal = 0;
    iEvent.getByToken(hgcalTowersToken, hgcalTowersHandle);
    for (auto &hit : *hgcalTowersHandle.product())
    {
        TowerHelper::TowerHit l1Hit;
        l1Hit.isBarrel     = false;
        l1Hit.towerEta     = hit.eta();
        l1Hit.towerPhi     = hit.phi();
        l1Hit.towerEm      = hit.etEm();
        l1Hit.towerHad     = hit.etHad();
        l1Hit.towerEt      = l1Hit.towerEm + l1Hit.towerHad;
        l1Hit.towerIeta    = endcap_ieta(l1Hit.towerEta);
        l1Hit.towerIphi    = endcap_iphi(l1Hit.towerPhi);
        l1Hit.towerIem     = floor( l1Hit.towerEm/0.5 );
        l1Hit.towerIhad    = floor( l1Hit.towerHad/0.5 );
        l1Hit.towerIet     = floor( (l1Hit.towerEm + l1Hit.towerHad)/0.5 );

        l1CaloTowers.push_back(l1Hit);

        if (l1Hit.towerIeta > maxIetaHGCal) { maxIetaHGCal = l1Hit.towerIeta; }
    }

    iEvent.getByToken(hcalDigisToken, hcalDigisHandle);
    const auto& decoder = eSetup.getData(decoderTag);
    for (const auto& hit : *hcalDigisHandle.product())
    {
        HcalTrigTowerDetId id = hit.id();
        
        // Only doing HF so skip outside range
        if (abs(id.ieta()) < l1t::CaloTools::kHFBegin) { continue; }
        if (abs(id.ieta()) > l1t::CaloTools::kHFEnd)   { continue; }    

        // get the energy deposit -> divide it by 2 to account fro iphi splitting
        float hadEt = decoder.hcaletValue(hit.id(), hit.t0()) / 2.;
        
        // shift HF ieta to fit the HGCAL towers
        int ietaShift = maxIetaHGCal - l1t::CaloTools::kHFBegin;

        TowerHelper::TowerHit l1Hit_A;
        l1Hit_A.isBarrel     = false;
        l1Hit_A.towerEta     = l1t::CaloTools::towerEta(id.ieta());
        l1Hit_A.towerPhi     = l1t::CaloTools::towerPhi(id.ieta(), id.iphi());
        l1Hit_A.towerEm      = 0.;
        l1Hit_A.towerHad     = hadEt;
        l1Hit_A.towerEt      = hadEt;
        l1Hit_A.towerIeta    = id.ieta() + ietaShift * std::copysign(1, l1Hit_A.towerEta);
        l1Hit_A.towerIphi    = id.iphi();
        l1Hit_A.towerIem     = 0;
        l1Hit_A.towerIhad    = floor( hadEt/0.5 );
        l1Hit_A.towerIet     = floor( hadEt/0.5 );

        TowerHelper::TowerHit l1Hit_B;
        l1Hit_B.isBarrel     = false;
        l1Hit_B.towerEta     = l1t::CaloTools::towerEta(id.ieta());
        l1Hit_B.towerPhi     = l1t::CaloTools::towerPhi(id.ieta(), id.iphi()) + 0.0872664; // account for iphi splitting
        l1Hit_B.towerEm      = 0.;
        l1Hit_B.towerHad     = hadEt;
        l1Hit_B.towerEt      = hadEt;
        l1Hit_B.towerIeta    = id.ieta() + ietaShift * std::copysign(1, l1Hit_B.towerEta);
        l1Hit_B.towerIphi    = id.iphi() + 1; // account for iphi splitting
        l1Hit_B.towerIem     = 0;
        l1Hit_B.towerIhad    = floor( hadEt/0.5 );
        l1Hit_B.towerIet     = floor( hadEt/0.5 );

        // the seeding happens only up to ieta 35 (endcap limit) so no need to store TT for higher than that
        if (abs(l1Hit_A.towerIeta)<=39) l1CaloTowers.push_back(l1Hit_A);
        if (abs(l1Hit_B.towerIeta)<=39) l1CaloTowers.push_back(l1Hit_B);
    }

    if (DEBUG)
    {
        std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b)
        {
            if (a.towerIeta == b.towerIeta) { return a.towerIphi < b.towerIphi; }
            else                            { return a.towerIeta < b.towerIeta; }
        });

        for (auto &l1CaloTower : l1CaloTowers)
        {
            printf("CALO TOWER iEta %i iPhi %i eta %f phi %f iem %i ihad %i iet %i nL1eg %i\n",
                (int)l1CaloTower.towerIeta,
                (int)l1CaloTower.towerIphi,
                l1CaloTower.towerEta,
                l1CaloTower.towerPhi,
                l1CaloTower.towerIem,
                l1CaloTower.towerIhad,
                l1CaloTower.towerIet,
                l1CaloTower.nL1eg);
        }
    }

    // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
    std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b) { return a.towerEt > b.towerEt; });

    if (DEBUG)
    {
        int n_towers = l1CaloTowers.size();
        int n_nonzero_towers = 0;
        for (auto &l1CaloTower : l1CaloTowers) { if (l1CaloTower.towerEt>0) n_nonzero_towers++; }
        std::cout << "***************************************************************************************************************************************" << std::endl;
        std::cout << " ** total number of towers = " << n_towers << std::endl;
        std::cout << " ** total number of non-zero towers = " << n_nonzero_towers << std::endl;
    }

    
    // Make NxM TowerClusters
    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClustersNxM(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for NxM TowerClusters seeds finding
    bool caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster cluNxM; cluNxM.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip HF towers for seeding
            if (abs(l1CaloTower.towerIeta) > 35) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (cluNxM.nHits == 0.0)
            {
                // the leading unused tower has ET < min, stop jet clustering
                if (l1CaloTower.towerEt < EtMinForSeeding)
                {
                    caloTauSeedingFinished = true;
                    break;
                }
                l1CaloTower.stale4seed = true;
                l1CaloTower.stale = true;

                // Set seed location
                if (l1CaloTower.isBarrel) { cluNxM.barrelSeeded = true; }
                
                // Fill the seed tower variables
                cluNxM.seedIeta = l1CaloTower.towerIeta;
                cluNxM.seedIphi = l1CaloTower.towerIphi;
                cluNxM.seedEta  = l1CaloTower.towerEta;
                cluNxM.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(cluNxM.seedIeta)<=13) { cluNxM.isBarrel = true;  }
                else if (abs(cluNxM.seedIeta)>=22) { cluNxM.isEndcap = true;  }
                else                               { cluNxM.isOverlap = true; }

                // Fill the TowerCluster towers
                cluNxM.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                cluNxM.totalEm      += l1CaloTower.towerEm;
                cluNxM.totalHad     += l1CaloTower.towerHad;
                cluNxM.totalEt      += l1CaloTower.towerEt;
                cluNxM.totalEgEt    += l1CaloTower.l1egTowerEt;
                cluNxM.totalIem     += l1CaloTower.towerIem;
                cluNxM.totalIhad    += l1CaloTower.towerIhad;
                cluNxM.totalIet     += l1CaloTower.towerIet;
                cluNxM.totalEgIet   += l1CaloTower.l1egTowerIet;
                cluNxM.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_dIEta(cluNxM.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_dIPhi(cluNxM.seedIphi, l1CaloTower.towerIphi);

            // stale tower for seeding if it would lead to overalp between clusters
            if (abs(d_iEta) <= (etaClusterDimension-1) && abs(d_iPhi) <= (phiClusterDimension-1)) { l1CaloTower.stale4seed = true; }
        } // end for loop over TPs

        if (cluNxM.nHits > 0) { l1TowerClustersNxM->push_back(cluNxM); }

    }  // end while loop of TowerClusters seeding

    // Create batch input for Tensorflow models
    tensorflow::setLogging("2");
    int batchSize =  (int)(l1TowerClustersNxM->size());
    tensorflow::TensorShape imageShape({batchSize, etaClusterDimension, phiClusterDimension, CNNfilters});
    tensorflow::TensorShape positionShape({batchSize, 2});
    tensorflow::Tensor TowerClusterImage(tensorflow::DT_FLOAT, imageShape);
    tensorflow::Tensor TowerClusterPosition(tensorflow::DT_FLOAT, positionShape);

    // loop for NxM TowerClusters creation starting from the seed just found
    int cluIdx = 0;
    for (auto& cluNxM : *l1TowerClustersNxM)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_dIEta(cluNxM.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_dIPhi(cluNxM.seedIphi, l1CaloTower.towerIphi);

            // cluster all towers in a NxM towers mask
            if (abs(d_iEta) <= (etaClusterDimension-1)/2 && abs(d_iPhi) <= (phiClusterDimension-1)/2)
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                cluNxM.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                cluNxM.totalEm      += l1CaloTower.towerEm;
                cluNxM.totalHad     += l1CaloTower.towerHad;
                cluNxM.totalEt      += l1CaloTower.towerEt;
                cluNxM.totalEgEt    += l1CaloTower.l1egTowerEt;
                cluNxM.totalIem     += l1CaloTower.towerIem;
                cluNxM.totalIhad    += l1CaloTower.towerIhad;
                cluNxM.totalIet     += l1CaloTower.towerIet;
                cluNxM.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) cluNxM.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLike(cluNxM.towerHits);
        cluNxM.InitHits(); cluNxM.towerHits = sortedHits;

        // Fill inputs for Tensorflow inference
        for (int eta = 0; eta < etaClusterDimension; ++eta)
        {
            for (int phi = 0; phi < phiClusterDimension; ++phi)
            {
                int towerIdx = eta*phiClusterDimension + phi - 1;
                if (CNNfilters == 3)
                {
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 0) = cluNxM.towerHits[towerIdx].l1egTowerEt;
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 1) = cluNxM.towerHits[towerIdx].towerEm;
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 2) = cluNxM.towerHits[towerIdx].towerHad;
                }
                else if (CNNfilters == 1)
                {
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 0) = cluNxM.towerHits[towerIdx].towerEt;
                }
            }
        }
        
        TowerClusterPosition.tensor<float, 2>()(cluIdx, 0) = cluNxM.seedEta;
        TowerClusterPosition.tensor<float, 2>()(cluIdx, 1) = cluNxM.seedPhi;
        
        cluIdx += 1; // increase index of cluster in batch

    }// end while loop of NxM TowerClusters creation

    // Apply CNN model
    tensorflow::NamedTensorList CNNinputList = {{"TowerClusterImage", TowerClusterImage}, {"TowerClusterPosition", TowerClusterPosition}};
    std::vector<tensorflow::Tensor> CNNoutputs;
    tensorflow::run(CNNsession, CNNinputList, {"middleMan"}, &CNNoutputs);
    tensorflow::NamedTensorList DNNinputsList = {{"middleMan", CNNoutputs[0]}};

    // Apply DNN for identification
    std::vector<tensorflow::Tensor> DNNoutputsIdent;
    tensorflow::run(DNNsessionIdent, DNNinputsList, {"sigmoid_DNNout"}, &DNNoutputsIdent);

    // Apply DNN for calibration
    std::vector<tensorflow::Tensor> DNNoutputsCalib;
    tensorflow::run(DNNsessionCalib, DNNinputsList, {"DNNout"}, &DNNoutputsCalib);

    // Fill CNN+DNN output variables of TowerClusters
    cluIdx = 0;
    for (auto& cluNxM : *l1TowerClustersNxM)
    {
        cluNxM.IDscore = DNNoutputsIdent[0].matrix<float>()(0, cluIdx);
        cluNxM.calibPt = DNNoutputsCalib[0].matrix<float>()(0, cluIdx);
        cluIdx += 1; // increase index of cluster in batch
    }

    if (DEBUG) 
    {
        std::cout << "\n***************************************************************************************************************************************" << std::endl;
        for (auto& cluNxM : *l1TowerClustersNxM)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu NxM seed " << " , eta " << cluNxM.seedIeta << " phi " << cluNxM.seedIphi << std::endl;
            std::cout << " -- clu NxM seed " << " , isBarrel " << cluNxM.isBarrel << " isEndcap " << cluNxM.isEndcap << " isOverlap " << cluNxM.isOverlap << std::endl;
            std::cout << " -- clu NxM towers etas (" << cluNxM.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < cluNxM.towerHits.size(); ++j) { std::cout  << ", " << cluNxM.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu NxM towers phis (" << cluNxM.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < cluNxM.towerHits.size(); ++j) { std::cout << ", " << cluNxM.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu NxM towers ems (" << cluNxM.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < cluNxM.towerHits.size(); ++j) { std::cout << ", " << cluNxM.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu NxM towers hads (" << cluNxM.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < cluNxM.towerHits.size(); ++j) { std::cout << ", " << cluNxM.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu NxM towers ets (" << cluNxM.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < cluNxM.towerHits.size(); ++j) { std::cout << ", " << cluNxM.towerHits[j].towerIet; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu NxM number of towers " << cluNxM.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;
    }

    // Create and Fill the collection of 3D clusters and their usefull attributes
    std::unique_ptr<HGClusterHelper::HGClustersCollection> HGClustersCollection(new HGClusterHelper::HGClustersCollection);
    iEvent.getByToken(HGClusterToken, HGClusterHandle);

    // Create batch input for XGBoost models
    const l1t::HGCalMulticlusterBxCollection& HGClusters = *HGClusterHandle;
    int nmb_cl3ds = HGClusters.size();
    int nmb_ident_feats = XGBident_feats.size();
    int nmb_calib_feats = XGBcalib_feats.size();
    float IdentData[nmb_cl3ds][nmb_ident_feats];
    float CalibData[nmb_cl3ds][nmb_calib_feats];

    cluIdx = 0;
    for (auto& cl3d : *HGClusterHandle.product())
    {
        HGClusterHelper::HGCluster HGCluster;

        HGCluster.pt = cl3d.pt();
        HGCluster.energy = cl3d.energy();
        HGCluster.eta = cl3d.eta();
        HGCluster.phi = cl3d.phi();
        HGCluster.showerlength = cl3d.showerLength();
        HGCluster.coreshowerlength = cl3d.coreShowerLength();
        HGCluster.firstlayer = cl3d.firstLayer();
        HGCluster.seetot = cl3d.sigmaEtaEtaTot();
        HGCluster.seemax = cl3d.sigmaEtaEtaMax();
        HGCluster.spptot = cl3d.sigmaPhiPhiTot();
        HGCluster.sppmax = cl3d.sigmaPhiPhiMax();
        HGCluster.szz = cl3d.sigmaZZ();
        HGCluster.srrtot = cl3d.sigmaRRTot();
        HGCluster.srrmax = cl3d.sigmaRRMax();
        HGCluster.srrmean = cl3d.sigmaRRMean();
        HGCluster.hoe = cl3d.hOverE();
        HGCluster.meanz = cl3d.zBarycenter();
        HGCluster.quality = cl3d.hwQual();

        for (int i = 0; i < nmb_ident_feats; ++i)
        {
            if (XGBident_feats[i] == "cl3d_pt")               { IdentData[cluIdx][i] =  HGCluster.pt; }
            if (XGBident_feats[i] == "cl3d_e")                { IdentData[cluIdx][i] =  HGCluster.energy; }
            if (XGBident_feats[i] == "cl3d_eta")              { IdentData[cluIdx][i] =  HGCluster.eta; }
            if (XGBident_feats[i] == "cl3d_abseta")           { IdentData[cluIdx][i] =  abs(HGCluster.eta); }
            if (XGBident_feats[i] == "cl3d_phi")              { IdentData[cluIdx][i] =  HGCluster.phi; }
            if (XGBident_feats[i] == "cl3d_showerlength")     { IdentData[cluIdx][i] =  HGCluster.showerlength; }
            if (XGBident_feats[i] == "cl3d_coreshowerlength") { IdentData[cluIdx][i] =  HGCluster.coreshowerlength; }
            if (XGBident_feats[i] == "cl3d_firstlayer")       { IdentData[cluIdx][i] =  HGCluster.firstlayer; }
            if (XGBident_feats[i] == "cl3d_seetot")           { IdentData[cluIdx][i] =  HGCluster.seetot; }
            if (XGBident_feats[i] == "cl3d_seemax")           { IdentData[cluIdx][i] =  HGCluster.seemax; }
            if (XGBident_feats[i] == "cl3d_spptot")           { IdentData[cluIdx][i] =  HGCluster.spptot; }
            if (XGBident_feats[i] == "cl3d_sppmax")           { IdentData[cluIdx][i] =  HGCluster.sppmax; }
            if (XGBident_feats[i] == "cl3d_szz")              { IdentData[cluIdx][i] =  HGCluster.szz; }
            if (XGBident_feats[i] == "cl3d_srrtot")           { IdentData[cluIdx][i] =  HGCluster.srrtot; }
            if (XGBident_feats[i] == "cl3d_srrmax")           { IdentData[cluIdx][i] =  HGCluster.srrmax; }
            if (XGBident_feats[i] == "cl3d_srrmean")          { IdentData[cluIdx][i] =  HGCluster.srrmean; }
            if (XGBident_feats[i] == "cl3d_hoe")              { IdentData[cluIdx][i] =  HGCluster.hoe; }
            if (XGBident_feats[i] == "cl3d_meanz")            { IdentData[cluIdx][i] =  HGCluster.meanz; }
            if (XGBident_feats[i] == "cl3d_quality")          { IdentData[cluIdx][i] =  HGCluster.quality; }
        }

        for (int i = 0; i < nmb_calib_feats; ++i)
        {
            if (XGBcalib_feats[i] == "cl3d_pt")               { CalibData[cluIdx][i] =  HGCluster.pt; }
            if (XGBcalib_feats[i] == "cl3d_e")                { CalibData[cluIdx][i] =  HGCluster.energy; }
            if (XGBcalib_feats[i] == "cl3d_eta")              { CalibData[cluIdx][i] =  HGCluster.eta; }
            if (XGBcalib_feats[i] == "cl3d_abseta")           { CalibData[cluIdx][i] =  abs(HGCluster.eta); }
            if (XGBcalib_feats[i] == "cl3d_phi")              { CalibData[cluIdx][i] =  HGCluster.phi; }
            if (XGBcalib_feats[i] == "cl3d_showerlength")     { CalibData[cluIdx][i] =  HGCluster.showerlength; }
            if (XGBcalib_feats[i] == "cl3d_coreshowerlength") { CalibData[cluIdx][i] =  HGCluster.coreshowerlength; }
            if (XGBcalib_feats[i] == "cl3d_firstlayer")       { CalibData[cluIdx][i] =  HGCluster.firstlayer; }
            if (XGBcalib_feats[i] == "cl3d_seetot")           { CalibData[cluIdx][i] =  HGCluster.seetot; }
            if (XGBcalib_feats[i] == "cl3d_seemax")           { CalibData[cluIdx][i] =  HGCluster.seemax; }
            if (XGBcalib_feats[i] == "cl3d_spptot")           { CalibData[cluIdx][i] =  HGCluster.spptot; }
            if (XGBcalib_feats[i] == "cl3d_sppmax")           { CalibData[cluIdx][i] =  HGCluster.sppmax; }
            if (XGBcalib_feats[i] == "cl3d_szz")              { CalibData[cluIdx][i] =  HGCluster.szz; }
            if (XGBcalib_feats[i] == "cl3d_srrtot")           { CalibData[cluIdx][i] =  HGCluster.srrtot; }
            if (XGBcalib_feats[i] == "cl3d_srrmax")           { CalibData[cluIdx][i] =  HGCluster.srrmax; }
            if (XGBcalib_feats[i] == "cl3d_srrmean")          { CalibData[cluIdx][i] =  HGCluster.srrmean; }
            if (XGBcalib_feats[i] == "cl3d_hoe")              { CalibData[cluIdx][i] =  HGCluster.hoe; }
            if (XGBcalib_feats[i] == "cl3d_meanz")            { CalibData[cluIdx][i] =  HGCluster.meanz; }
            if (XGBcalib_feats[i] == "cl3d_quality")          { CalibData[cluIdx][i] =  HGCluster.quality; }
        }

        if (DEBUG)
        {
            std::cout << "---------------------------------------------------------------------" <<std::endl;
            std::cout << "pt :               in " << cl3d.pt()               << " out " << HGCluster.pt               << std::endl;
            std::cout << "energy :           in " << cl3d.energy()           << " out " << HGCluster.energy           << std::endl;
            std::cout << "eta :              in " << cl3d.eta()              << " out " << HGCluster.eta              << std::endl;
            std::cout << "phi :              in " << cl3d.phi()              << " out " << HGCluster.phi              << std::endl;
            std::cout << "showerlength :     in " << cl3d.showerLength()     << " out " << HGCluster.showerlength     << std::endl;
            std::cout << "coreshowerlength : in " << cl3d.coreShowerLength() << " out " << HGCluster.coreshowerlength << std::endl;
            std::cout << "firstlayer :       in " << cl3d.firstLayer()       << " out " << HGCluster.firstlayer       << std::endl;
            std::cout << "seetot :           in " << cl3d.sigmaEtaEtaTot()   << " out " << HGCluster.seetot           << std::endl;
            std::cout << "seemax :           in " << cl3d.sigmaEtaEtaMax()   << " out " << HGCluster.seemax           << std::endl;
            std::cout << "spptot :           in " << cl3d.sigmaPhiPhiTot()   << " out " << HGCluster.spptot           << std::endl;
            std::cout << "sppmax :           in " << cl3d.sigmaPhiPhiMax()   << " out " << HGCluster.sppmax           << std::endl;
            std::cout << "szz :              in " << cl3d.sigmaZZ()          << " out " << HGCluster.szz              << std::endl;
            std::cout << "srrtot :           in " << cl3d.sigmaRRTot()       << " out " << HGCluster.srrtot           << std::endl;
            std::cout << "srrmax :           in " << cl3d.sigmaRRMax()       << " out " << HGCluster.srrmax           << std::endl;
            std::cout << "srrmean :          in " << cl3d.sigmaRRMean()      << " out " << HGCluster.srrmean          << std::endl;
            std::cout << "hoe :              in " << cl3d.hOverE()           << " out " << HGCluster.hoe              << std::endl;
            std::cout << "meanz :            in " << cl3d.zBarycenter()      << " out " << HGCluster.meanz            << std::endl;
            std::cout << "quality :          in " << cl3d.hwQual()           << " out " << HGCluster.quality          << std::endl;
            std::cout << "---------------------------------------------------------------------" <<std::endl;
        }

        HGClustersCollection->push_back(HGCluster);
        cluIdx += 1; // increase index of cluster in batch
    }

    // Apply XGB for identification
    DMatrixHandle IdentMatrix;
    XGDMatrixCreateFromMat((float *)IdentData,nmb_cl3ds,nmb_ident_feats,-1,&IdentMatrix);
    bst_ulong LENoutputsIdent = 0;
    const float *XGBoutputsIdent;
    auto ret1=XGBoosterPredict(XGBident, IdentMatrix,0, 0,0,&LENoutputsIdent,&XGBoutputsIdent);

    // Apply XGB for calibration
    DMatrixHandle CalibMatrix;
    XGDMatrixCreateFromMat((float *)CalibData,nmb_cl3ds,nmb_ident_feats,-1,&CalibMatrix);
    bst_ulong LENoutputsCalib = 0;
    const float *XGBoutputsCalib;
    auto ret2=XGBoosterPredict(XGBcalib, CalibMatrix,0, 0,0,&LENoutputsCalib,&XGBoutputsCalib);

    cluIdx = 0;
    for (auto& HGCluster : *HGClustersCollection)
    {
        HGCluster.IDscore = XGBoutputsIdent[cluIdx];
        
        float c1pt = HGCluster.pt + (C1calib_params[0] * HGCluster.pt + C1calib_params[1]);
        float c2pt = c1pt * XGBoutputsCalib[cluIdx];
        float c2pt_log = log(abs(c2pt));
        float c3pt = c2pt / (C3calib_params[0] + C3calib_params[1] * c2pt_log + C3calib_params[2] * pow(c2pt_log,2) + C3calib_params[3] * pow(c2pt_log,3) + C3calib_params[4] * pow(c2pt_log,4));
        HGCluster.calibPt = c3pt;
        
        cluIdx += 1; // increase index of cluster in batch
    }

    // Create and Fill the collection of L1 taus and their usefull attributes
    std::unique_ptr<TauHelper::TausCollection> TausCollection(new TauHelper::TausCollection);

    // at the same time: cross loop over NxM TowerClusters and CL3D to do matching in the endcap
    int cluNxMIdx = -1;
    for (auto& cluNxM : *l1TowerClustersNxM)
    {
        cluNxMIdx += 1;
        if (cluNxM.IDscore<0/*FIXME*/) { continue; } // consider only clusters that pass the minimal 99% efficiency cut

        TauHelper::Tau Tau;

        // treat endcap and barrel separartely
        if (abs(cluNxM.seedIeta>15)){
            int matchedCluIdx = -99;
            float dR2min = 0.2225; // set min dR at 0.47^2 = 0.25^2 + 0.4^2
            int cl3dIdx = -1;
            for (auto& HGCluster : *HGClustersCollection)
            {
                cl3dIdx += 1;
                if (HGCluster.IDscore<0/*FIXME*/) { continue; } // consider only clusters that pass the minimal 99% efficiency cut

                // apply geometrical match between cluNxM and cl3d
                float dEta = cluNxM.seedEta - HGCluster.eta;
                float dPhi = reco::deltaPhi(cluNxM.seedPhi, HGCluster.phi);
                if (dEta > 0.25 && dPhi > 0.4) { continue; } //FIXME

                float dR2 = dEta * dEta + dPhi * dPhi;
                if (dR2 <= dR2min)
                {
                    dR2min = dR2;
                    matchedCluIdx = cl3dIdx;
                }
                if (matchedCluIdx != -99)
                {
                    // set tau information for the endcap area
                    Tau.pt  = HGCluster.calibPt;
                    Tau.eta = HGCluster.eta;
                    Tau.phi = HGCluster.phi;
                    Tau.clusterIdx = matchedCluIdx;
                    Tau.isEndcap = true;
                    Tau.IDscore = HGCluster.IDscore;
                }
            }
        }
        else
        {
            // set tau information for the barrel area
            Tau.pt  = cluNxM.calibPt;
            Tau.eta = cluNxM.seedEta;
            Tau.phi = cluNxM.seedPhi;
            Tau.clusterIdx = cluNxMIdx;
            Tau.isBarrel = true;
            Tau.IDscore = cluNxM.IDscore;
        }

        TausCollection->push_back(Tau);
    }

    iEvent.put(std::move(l1TowerClustersNxM), "l1TowerClustersNxM");
    iEvent.put(std::move(HGClustersCollection), "HGClustersCollection");
    iEvent.put(std::move(TausCollection), "TausCollection");
}

int L1CaloTauProducer::tower_dIPhi(int &iPhi_1, int &iPhi_2) const
{
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    if (result > PI)   { result -= 2 * PI; }
    if (result <= -PI) { result += 2 * PI; } 
    return result;
}

int L1CaloTauProducer::tower_dIEta(int &iEta_1, int &iEta_2) const
{
    if (iEta_1 * iEta_2 > 0) { return iEta_1 - iEta_2; }
    else
    {
        if (iEta_1>0) { return iEta_1 - iEta_2 - 1; }
        else          { return iEta_1 - iEta_2 + 1; }
    }
}

int L1CaloTauProducer::endcap_iphi(float &phi) const
{
    float phi_step = 0.0872664;
    if (phi > 0) { return floor(phi / phi_step) + 1;  }
    else         { return floor(phi / phi_step) + 73; }
}

int L1CaloTauProducer::endcap_ieta(float &eta) const
{
    float eta_step = 0.0845;
    return floor(abs(eta)/eta_step) * std::copysign(1,eta);
}

std::vector<TowerHelper::TowerHit> L1CaloTauProducer::sortPicLike(std::vector<TowerHelper::TowerHit> towerHits) const
{
    // sorts towers in order of eta,phi (both increasing)
    // e.g. 3x5 cluster : eta 1,1,1,1,1,2,2,2,2,2,3,3,3,3,3
    //                    phi 1,2,3,4,5,1,2,3,4,5,1,2,3,4,5
    std::sort(begin(towerHits), end(towerHits), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b)
    { 
        if (a.towerIeta == b.towerIeta)
        {
            if (((a.towerIphi>=65 && a.towerIphi<=72) && (b.towerIphi>=1 && b.towerIphi<=8)) ||
                ((b.towerIphi>=65 && b.towerIphi<=72) && (a.towerIphi>=1 && a.towerIphi<=8)))   { return a.towerIphi > b.towerIphi; }
            
            else { return a.towerIphi < b.towerIphi; }
        }
        else { return a.towerIeta < b.towerIeta; }
    });

    return towerHits;
}

DEFINE_FWK_MODULE(L1CaloTauProducer);