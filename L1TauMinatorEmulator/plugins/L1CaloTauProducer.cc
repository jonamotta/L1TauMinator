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
        float inputQuantizer(float inputF, float LSB, int nbits);

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

        std::string CLTW_CNNmodel_path;
        std::string CLTW_DNNident_path;
        std::string CLTW_DNNcalib_path;

        std::string CL3D_DNNident_path;
        std::string CL3D_DNNcalib_path;

        std::vector<std::string> CL3D_DNNident_feats;
        std::vector<std::string> CL3D_DNNcalib_feats;

        bool DEBUG;

        tensorflow::GraphDef* CLTW_CNNmodel;
        tensorflow::GraphDef* CLTW_DNNident;
        tensorflow::GraphDef* CLTW_DNNcalib;

        tensorflow::Session* CLTW_CNNsession;
        tensorflow::Session* CLTW_DNNsessionIdent;
        tensorflow::Session* CLTW_DNNsessionCalib;

        tensorflow::GraphDef* CL3D_DNNident;
        tensorflow::GraphDef* CL3D_DNNcalib;

        tensorflow::Session* CL3D_DNNsessionIdent;
        tensorflow::Session* CL3D_DNNsessionCalib;
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
      CLTW_CNNmodel_path(iConfig.getParameter<std::string>("CLTW_CNNmodel_path")),
      CLTW_DNNident_path(iConfig.getParameter<std::string>("CLTW_DNNident_path")),
      CLTW_DNNcalib_path(iConfig.getParameter<std::string>("CLTW_DNNcalib_path")),
      CL3D_DNNident_path(iConfig.getParameter<std::string>("CL3D_DNNident_path")),
      CL3D_DNNcalib_path(iConfig.getParameter<std::string>("CL3D_DNNcalib_path")),
      CL3D_DNNident_feats(iConfig.getParameter<std::vector<std::string>>("CL3D_DNNident_feats")),
      CL3D_DNNcalib_feats(iConfig.getParameter<std::vector<std::string>>("CL3D_DNNcalib_feats")),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    

    // Create sessions for Tensorflow inferece
    CLTW_CNNmodel = tensorflow::loadGraphDef(CLTW_CNNmodel_path);
    CLTW_CNNsession = tensorflow::createSession(CLTW_CNNmodel);

    CLTW_DNNident = tensorflow::loadGraphDef(CLTW_DNNident_path);
    CLTW_DNNsessionIdent = tensorflow::createSession(CLTW_DNNident);

    CLTW_DNNcalib = tensorflow::loadGraphDef(CLTW_DNNcalib_path);
    CLTW_DNNsessionCalib = tensorflow::createSession(CLTW_DNNcalib);

    CL3D_DNNident = tensorflow::loadGraphDef(CL3D_DNNident_path);
    CL3D_DNNsessionIdent = tensorflow::createSession(CL3D_DNNident);

    CL3D_DNNcalib = tensorflow::loadGraphDef(CL3D_DNNcalib_path);
    CL3D_DNNsessionCalib = tensorflow::createSession(CL3D_DNNcalib);


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
    int warnings = 0;
    for (auto &hit : *l1CaloTowerHandle.product())
    {
        if (hit.towerIEta() == -1016 && hit.towerIPhi() == -962) { warnings += 1; }

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
    if (warnings != 0) { std::cout << " ** WARNING : FOUND " << warnings << " TOWERS WITH towerIeta=-1016 AND towerIphi=-962" << std::endl; }

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

    // Fill missing towers in HGCAL due to current simulation limitation (because right now they are produced starting from modules and some TT do not have a centered module in front of them)
    // --> just fill with zeros (JB approves)
    std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b)
    {
        if (a.towerIeta == b.towerIeta) { return a.towerIphi < b.towerIphi; }
        else                            { return a.towerIeta < b.towerIeta; }
    });
    int iEtaProgression = -39;
    int iPhiProgression = 1;
    for (auto &l1CaloTower : l1CaloTowers)
    {
        // skip towers with the weird IEta=-1016 IPhi=-962 values and they will also be filled with zeros
        if (l1CaloTower.towerIeta < -39) { continue; }

        int iPhiDiff = l1CaloTower.towerIphi - iPhiProgression;
        int max = iPhiProgression+iPhiDiff;
        if (iPhiDiff != 0)
        {   
            if (iPhiDiff<0) { max = 73; } // account for circularity 72->1
            for (int iPhi=iPhiProgression; iPhi < max; ++iPhi)
            {
                TowerHelper::TowerHit l1Hit;
                if (abs(iEtaProgression)<=17) { l1Hit.isBarrel = true; }
                else                          { l1Hit.isBarrel = false; }
                l1Hit.l1egTowerEt  = 0;
                l1Hit.l1egTowerIet = 0;
                l1Hit.nL1eg        = 0;
                l1Hit.towerEta     = iEtaProgression * 0.0845;
                if (iPhi<=36) { l1Hit.towerPhi = 0.043633 + (iPhi-1) * 0.0872664; }
                else          { l1Hit.towerPhi = 0.043633 + (iPhi-72-1) * 0.0872664; }
                l1Hit.towerEm      = 0.0;
                l1Hit.towerHad     = 0.0;
                l1Hit.towerEt      = 0.0;
                l1Hit.towerIeta    = iEtaProgression;
                l1Hit.towerIphi    = iPhi;
                l1Hit.towerIem     = 0;
                l1Hit.towerIhad    = 0;
                l1Hit.towerIet     = 0;
                
                l1CaloTowers.push_back(l1Hit);

                if (DEBUG) { std::cout << "Adding missing tower with iEta " << iEtaProgression << " iPhi " << iPhi << std::endl; }

                iPhiProgression += 1;
            }
            if (max==73) // hack to account for circularity 72->1
            {
                iEtaProgression += 1;
                if (iEtaProgression == 0) { iEtaProgression += 1;} // skip iEta=0
                
                for (int iPhi=1; iPhi < l1CaloTower.towerIphi; ++iPhi)
                {
                    TowerHelper::TowerHit l1Hit;
                    if (abs(iEtaProgression)<=17) { l1Hit.isBarrel = true; }
                    else                          { l1Hit.isBarrel = false; }
                    l1Hit.l1egTowerEt  = 0;
                    l1Hit.l1egTowerIet = 0;
                    l1Hit.nL1eg        = 0;
                    l1Hit.towerEta     = iEtaProgression * 0.0845;
                    if (iPhi<=36) { l1Hit.towerPhi = 0.043633 + (iPhi-1) * 0.0872664; }
                    else          { l1Hit.towerPhi = 0.043633 + (iPhi-72-1) * 0.0872664; }
                    l1Hit.towerEm      = 0.0;
                    l1Hit.towerHad     = 0.0;
                    l1Hit.towerEt      = 0.0;
                    l1Hit.towerIeta    = iEtaProgression;
                    l1Hit.towerIphi    = iPhi;
                    l1Hit.towerIem     = 0;
                    l1Hit.towerIhad    = 0;
                    l1Hit.towerIet     = 0;
                    
                    l1CaloTowers.push_back(l1Hit);

                    if (DEBUG) { std::cout << "Adding missing tower with iEta " << iEtaProgression << " iPhi " << iPhi << std::endl; }

                    iPhiProgression += 1;
                }

                iEtaProgression -= 1;
                if (iEtaProgression == 0) { iEtaProgression -= 1;} // skip iEta=0
            }
        }
        
        iPhiProgression += 1;
        if (iPhiProgression > 72)
        {
            iPhiProgression = 1;
            iEtaProgression += 1;
            if (iEtaProgression == 0) { iEtaProgression += 1;} // skip iEta=0
        }
        if (max == 73) { iPhiProgression = l1CaloTower.towerIphi+1; } // hack to account for circularity 72->1

        if (iEtaProgression>39) { break; }
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
            printf("CALO TOWER iEta %i iPhi %i eta %f phi %f em %f had %f et %f nL1eg %i\n",
                (int)l1CaloTower.towerIeta,
                (int)l1CaloTower.towerIphi,
                l1CaloTower.towerEta,
                l1CaloTower.towerPhi,
                l1CaloTower.towerEm,
                l1CaloTower.towerHad,
                l1CaloTower.towerEt,
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

        if (int(sortedHits.size()) != etaClusterDimension*phiClusterDimension) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

        // Fill inputs for Tensorflow inference
        for (int eta = 0; eta < etaClusterDimension; ++eta)
        {
            for (int phi = 0; phi < phiClusterDimension; ++phi)
            {
                int towerIdx = eta*phiClusterDimension + phi;
                if (CNNfilters == 3)
                {
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 0) = inputQuantizer(cluNxM.towerHits[towerIdx].l1egTowerEt, 0.25, 10);
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 1) = inputQuantizer(cluNxM.towerHits[towerIdx].towerEm, 0.25, 10);
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 2) = inputQuantizer(cluNxM.towerHits[towerIdx].towerHad, 0.25, 10);
                }
                else if (CNNfilters == 1)
                {
                    TowerClusterImage.tensor<float, 4>()(cluIdx, eta, phi, 0) = cluNxM.towerHits[towerIdx].towerEt;
                }

                if (DEBUG)
                {
                    std::cout << "(" << eta << "," << phi << ")[" << towerIdx << "]        " << cluNxM.towerHits[towerIdx].l1egTowerEt << "    " << cluNxM.towerHits[towerIdx].towerEm << "    " << cluNxM.towerHits[towerIdx].towerHad << "\n" << std::endl;
                    if (phi==phiClusterDimension-1) { std::cout << "" << std::endl; }
                }
            }
        }
        
        TowerClusterPosition.tensor<float, 2>()(cluIdx, 0) = cluNxM.seedEta;
        TowerClusterPosition.tensor<float, 2>()(cluIdx, 1) = cluNxM.seedPhi;
        
        cluIdx += 1; // increase index of cluster in batch

    }// end while loop of NxM TowerClusters creation

    // Apply CNN model
    tensorflow::NamedTensorList CLTW_CNNinputList = {{"TowerClusterImage", TowerClusterImage}, {"TowerClusterPosition", TowerClusterPosition}};
    std::vector<tensorflow::Tensor> CLTW_CNNoutputs;
    tensorflow::run(CLTW_CNNsession, CLTW_CNNinputList, {"model/middleMan/concat"}, &CLTW_CNNoutputs);
    tensorflow::NamedTensorList CLTW_DNNinputsList = {{"middleMan", CLTW_CNNoutputs[0]}};

    // Apply DNN for identification
    std::vector<tensorflow::Tensor> CLTW_DNNoutputsIdent;
    tensorflow::run(CLTW_DNNsessionIdent, CLTW_DNNinputsList, {"model_1/sigmoid_DNNout/Sigmoid"}, &CLTW_DNNoutputsIdent);

    // Apply DNN for calibration
    std::vector<tensorflow::Tensor> CLTW_DNNoutputsCalib;
    tensorflow::run(CLTW_DNNsessionCalib, CLTW_DNNinputsList, {"TauCNNCalibrator/DNNout/MatMul"}, &CLTW_DNNoutputsCalib);

    // Fill CNN+DNN output variables of TowerClusters
    cluIdx = 0;
    for (auto& cluNxM : *l1TowerClustersNxM)
    {
        cluNxM.IDscore = CLTW_DNNoutputsIdent[0].matrix<float>()(0, cluIdx);
        cluNxM.calibPt = CLTW_DNNoutputsCalib[0].matrix<float>()(0, cluIdx);
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

    // Create batch input for Tensorflow models
    tensorflow::setLogging("2");
    batchSize =  (int)(HGClustersCollection->size());
    const int nmb_ident_feats = CL3D_DNNident_feats.size();
    const int nmb_calib_feats = CL3D_DNNcalib_feats.size();
    tensorflow::TensorShape identInputShape({batchSize, nmb_ident_feats});
    tensorflow::Tensor Cl3dIdentInput(tensorflow::DT_FLOAT, identInputShape);
    tensorflow::TensorShape calibInputShape({batchSize, nmb_calib_feats});
    tensorflow::Tensor Cl3dCalibInput(tensorflow::DT_FLOAT, calibInputShape);

    cluIdx = 0;
    for (auto& HGCluster : *HGClustersCollection)
    {
        for (int i = 0; i < nmb_ident_feats; ++i)
        {
            if (CL3D_DNNident_feats[i] == "cl3d_pt")               { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.pt, 0.25, 14) - 20.241386991916283)/23.01837359563478; }
            if (CL3D_DNNident_feats[i] == "cl3d_e")                { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.energy, 0.25, 14); }
            if (CL3D_DNNident_feats[i] == "cl3d_eta")              { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  HGCluster.eta; }
            if (CL3D_DNNident_feats[i] == "cl3d_localAbsEta")      { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(abs(HGCluster.eta)-1.45, 0.004, 9) - 1.0074996364465862)/0.49138567390141064; }
            if (CL3D_DNNident_feats[i] == "cl3d_phi")              { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  HGCluster.phi; }
            if (CL3D_DNNident_feats[i] == "cl3d_showerlength")     { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (HGCluster.showerlength - 35.80472904577922)/7.440518440086627; }
            if (CL3D_DNNident_feats[i] == "cl3d_coreshowerlength") { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (HGCluster.coreshowerlength - 11.933355669294706)/4.791972733791438; }
            if (CL3D_DNNident_feats[i] == "cl3d_firstlayer")       { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (HGCluster.firstlayer - 1.3720078128341484)/1.6893861207515388; }
            if (CL3D_DNNident_feats[i] == "cl3d_seetot")           { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.seetot, 0.0000153, 16) - 0.03648582652086512)/0.020407089049271552; }
            if (CL3D_DNNident_feats[i] == "cl3d_seemax")           { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.seemax, 0.0000153, 16); }
            if (CL3D_DNNident_feats[i] == "cl3d_spptot")           { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.spptot, 0.0000153, 16); }
            if (CL3D_DNNident_feats[i] == "cl3d_sppmax")           { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.sppmax, 0.0000153, 16); }
            if (CL3D_DNNident_feats[i] == "cl3d_szz")              { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.szz, 0.002, 16) - 20.51621627863874)/11.633317917896875; }
            if (CL3D_DNNident_feats[i] == "cl3d_srrtot")           { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.srrtot, 0.0000153, 16) - 0.00534390307737272)/0.001325129860675611; }
            if (CL3D_DNNident_feats[i] == "cl3d_srrmax")           { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.srrmax, 0.0000153, 16); }
            if (CL3D_DNNident_feats[i] == "cl3d_srrmean")          { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.srrmean, 0.0000153, 16) - 0.00365570411813347367)/0.0009327963551387752; }
            if (CL3D_DNNident_feats[i] == "cl3d_hoe")              { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.hoe, 0.002, 16) - 1.3676566630073708)/7.978238945457623; }
            if (CL3D_DNNident_feats[i] == "cl3d_localAbsMeanZ")    { Cl3dIdentInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(10*(abs(HGCluster.meanz)-320), 0.5, 12) - 291.6762877632198)/178.8235004591792; }
        }

        for (int i = 0; i < nmb_calib_feats; ++i)
        {
            if (CL3D_DNNcalib_feats[i] == "cl3d_pt")               { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.pt, 0.25, 14) - 20.241386991916283)/23.01837359563478; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_e")                { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.energy, 0.25, 14); }
            if (CL3D_DNNcalib_feats[i] == "cl3d_eta")              { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  HGCluster.eta; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_localAbsEta")      { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(abs(HGCluster.eta)-1.45, 0.004, 9) - 1.0074996364465862)/0.49138567390141064; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_phi")              { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  HGCluster.phi; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_showerlength")     { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (HGCluster.showerlength - 35.80472904577922)/7.440518440086627; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_coreshowerlength") { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (HGCluster.coreshowerlength - 11.933355669294706)/4.791972733791438; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_firstlayer")       { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (HGCluster.firstlayer - 1.3720078128341484)/1.6893861207515388; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_seetot")           { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.seetot, 0.0000153, 16) - 0.03648582652086512)/0.020407089049271552; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_seemax")           { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.seemax, 0.0000153, 16); }
            if (CL3D_DNNcalib_feats[i] == "cl3d_spptot")           { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.spptot, 0.0000153, 16); }
            if (CL3D_DNNcalib_feats[i] == "cl3d_sppmax")           { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.sppmax, 0.0000153, 16); }
            if (CL3D_DNNcalib_feats[i] == "cl3d_szz")              { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.szz, 0.002, 16) - 20.51621627863874)/11.633317917896875; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_srrtot")           { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.srrtot, 0.0000153, 16) - 0.00534390307737272)/0.001325129860675611; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_srrmax")           { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  inputQuantizer(HGCluster.srrmax, 0.0000153, 16); }
            if (CL3D_DNNcalib_feats[i] == "cl3d_srrmean")          { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.srrmean, 0.0000153, 16) - 0.00365570411813347367)/0.0009327963551387752; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_hoe")              { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(HGCluster.hoe, 0.002, 16) - 1.3676566630073708)/7.978238945457623; }
            if (CL3D_DNNcalib_feats[i] == "cl3d_localAbsMeanZ")    { Cl3dCalibInput.tensor<float, 2>()(cluIdx, i) =  (inputQuantizer(10*(abs(HGCluster.meanz)-320), 0.5, 12) - 291.6762877632198)/178.8235004591792; }
        }

        cluIdx += 1; // increase index of cluster in batch
    }

    // Apply DNN for identification
    tensorflow::NamedTensorList CL3D_DNNIdentInputsList = {{"CL3DFeatures", Cl3dIdentInput}};
    std::vector<tensorflow::Tensor> CL3D_DNNoutputsIdent;
    tensorflow::run(CL3D_DNNsessionIdent, CL3D_DNNIdentInputsList, {"TauDNNIdentifier/sigmoid_DNNout/Sigmoid"}, &CL3D_DNNoutputsIdent);

    // Apply DNN for calibration
    tensorflow::NamedTensorList CL3D_DNNCalibInputsList = {{"CL3DFeatures", Cl3dCalibInput}};
    std::vector<tensorflow::Tensor> CL3D_DNNoutputsCalib;
    tensorflow::run(CL3D_DNNsessionCalib, CL3D_DNNCalibInputsList, {"TauDNNCalibrator/DNNout/MatMul"}, &CL3D_DNNoutputsCalib);

    // Fill CNN+DNN output variables of TowerClusters
    cluIdx = 0;
    for (auto& HGCluster : *HGClustersCollection)
    {
        HGCluster.IDscore = CL3D_DNNoutputsIdent[0].matrix<float>()(0, cluIdx);
        HGCluster.calibPt = CL3D_DNNoutputsCalib[0].matrix<float>()(0, cluIdx);
        cluIdx += 1; // increase index of cluster in batch
    }

    // Create and Fill the collection of L1 taus and their usefull attributes
    std::unique_ptr<TauHelper::TausCollection> TausCollection(new TauHelper::TausCollection);

    // cross loop over NxM TowerClusters and CL3D to do matching in the endcap
    for (auto& cluNxM : *l1TowerClustersNxM)
    {
        if (cluNxM.IDscore<0/*FIXME*/) { continue; } // consider only clusters that pass the minimal 99% efficiency cut

        TauHelper::Tau Tau;

        // treat endcap and barrel separartely
        if (abs(cluNxM.seedEta)>1.5)
        {
            bool matched = false;
            float IDmax = 0.0;
            HGClusterHelper::HGCluster HGCluster2store; 
            for (auto& HGCluster : *HGClustersCollection)
            {
                if (HGCluster.IDscore<0/*FIXME*/) { continue; } // consider only clusters that pass the minimal 99% efficiency cut

                // apply geometrical match between cluNxM and cl3d
                float dEta = cluNxM.seedEta - HGCluster.eta;
                float dPhi = reco::deltaPhi(cluNxM.seedPhi, HGCluster.phi);
                float dR2 = dEta * dEta + dPhi * dPhi;
                if (dR2 > 0.25) { continue; } // require the cluster to be within dR 0.5 

                if (HGCluster.IDscore > IDmax)
                {
                    IDmax = HGCluster.IDscore;
                    HGCluster2store = HGCluster;
                    matched = true;
                }
            }
            if (matched)
            {
                // set tau information for the endcap area
                Tau.pt  = HGCluster2store.calibPt;
                Tau.eta = HGCluster2store.eta;
                Tau.phi = HGCluster2store.phi;
                Tau.isEndcap = true;
                Tau.IDscore = HGCluster2store.IDscore;
            }
        }
        else
        {
            // set tau information for the barrel area
            Tau.pt  = cluNxM.calibPt;
            Tau.eta = cluNxM.seedEta;
            Tau.phi = cluNxM.seedPhi;
            Tau.isBarrel = true;
            Tau.IDscore = cluNxM.IDscore;
        }

        if (Tau.isBarrel || Tau.isEndcap) { TausCollection->push_back(Tau); }
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

float L1CaloTauProducer::inputQuantizer(float inputF, float LSB, int nbits)
{
    return min( floor(inputF/LSB), float(pow(2,nbits)-1) ) * LSB;
}


DEFINE_FWK_MODULE(L1CaloTauProducer);