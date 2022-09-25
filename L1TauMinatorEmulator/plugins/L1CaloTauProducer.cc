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
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"

class L1CaloTauProducer : public edm::stream::EDProducer<> {
    public:
        explicit L1CaloTauProducer(const edm::ParameterSet&);

    private:
        //----edm control---
        void produce(edm::Event&, const edm::EventSetup&) override;

        //----private functions----
        int tower_diPhi(int &iPhi_1, int &iPhi_2) const;
        int tower_diEta(int &iEta_1, int &iEta_2) const;
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
        double EcalEtMinForClustering;
        double HcalEtMinForClustering;
        double EtMinForSeeding;
        bool DEBUG;
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
      HGClusterToken(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalClusters"))),
      decoderTag(esConsumes<CaloTPGTranscoder, CaloTPGRecord>(edm::ESInputTag("", ""))),
      EcalEtMinForClustering(iConfig.getParameter<double>("EcalEtMinForClustering")),
      HcalEtMinForClustering(iConfig.getParameter<double>("HcalEtMinForClustering")),
      EtMinForSeeding(iConfig.getParameter<double>("EtMinForSeeding")),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters9x9");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters7x7");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters5x5");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters5x9");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters5x7");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters3x7");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters3x5");

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

    
    // Make 5x9 TowerClusters
    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters5x9(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 5x9 TowerClusters seeds finding
    caloJetSeedingFinished = false;
    while (!caloJetSeedingFinished)
    {
        TowerHelper::TowerCluster clu5x9; clu5x9.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip HF towers for seeding
            if (abs(l1CaloTower.towerIeta) > 35) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu5x9.nHits == 0.0)
            {
                // the leading unused tower has ET < min, stop jet clustering
                if (l1CaloTower.towerEt < EtMinForSeeding)
                {
                    caloJetSeedingFinished = true;
                    break;
                }
                l1CaloTower.stale4seed = true;
                l1CaloTower.stale = true;

                // Set seed location
                if (l1CaloTower.isBarrel) { clu5x9.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu5x9.seedIeta = l1CaloTower.towerIeta;
                clu5x9.seedIphi = l1CaloTower.towerIphi;
                clu5x9.seedEta  = l1CaloTower.towerEta;
                clu5x9.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu5x9.seedIeta)<=13) { clu5x9.isBarrel = true;  }
                else if (abs(clu5x9.seedIeta)>=22) { clu5x9.isEndcap = true;  }
                else                               { clu5x9.isOverlap = true; }

                // Fill the TowerCluster towers
                clu5x9.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu5x9.totalEm      += l1CaloTower.towerEm;
                clu5x9.totalHad     += l1CaloTower.towerHad;
                clu5x9.totalEt      += l1CaloTower.towerEt;
                clu5x9.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu5x9.totalIem     += l1CaloTower.towerIem;
                clu5x9.totalIhad    += l1CaloTower.towerIhad;
                clu5x9.totalIet     += l1CaloTower.towerIet;
                clu5x9.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu5x9.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu5x9.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu5x9.seedIphi, l1CaloTower.towerIphi);

            // stale tower for seeding if it would lead to overalp between clusters
            if (abs(d_iEta) <= 4 && abs(d_iPhi) <= 8) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu5x9.nHits > 0.0) { l1TowerClusters5x9->push_back(clu5x9); }

    }  // end while loop of TowerClusters seeding

    // loop for 5x9 TowerClusters creation starting from the seed just found
    for (auto& clu5x9 : *l1TowerClusters5x9)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu5x9.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu5x9.seedIphi, l1CaloTower.towerIphi);

            // cluster all towers in a 5x9 towers mask
            if (abs(d_iEta) <= 2 && abs(d_iPhi) <= 4)
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu5x9.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu5x9.totalEm      += l1CaloTower.towerEm;
                clu5x9.totalHad     += l1CaloTower.towerHad;
                clu5x9.totalEt      += l1CaloTower.towerEt;
                clu5x9.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu5x9.totalIem     += l1CaloTower.towerIem;
                clu5x9.totalIhad    += l1CaloTower.towerIhad;
                clu5x9.totalIet     += l1CaloTower.towerIet;
                clu5x9.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu5x9.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLike(clu5x9.towerHits);
        clu5x9.InitHits(); clu5x9.towerHits = sortedHits;

    }// end while loop of 5x9 TowerClusters creation

    if (DEBUG) 
    {
        std::cout << "\n***************************************************************************************************************************************" << std::endl;
        for (auto& clu5x9 : *l1TowerClusters5x9)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 5x9 seed " << " , eta " << clu5x9.seedIeta << " phi " << clu5x9.seedIphi << std::endl;
            std::cout << " -- clu 5x9 seed " << " , isBarrel " << clu5x9.isBarrel << " isEndcap " << clu5x9.isEndcap << " isOverlap " << clu5x9.isOverlap << std::endl;
            std::cout << " -- clu 5x9 towers etas (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout  << ", " << clu5x9.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers phis (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout << ", " << clu5x9.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers ems (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout << ", " << clu5x9.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers hads (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout << ", " << clu5x9.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers ets (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout << ", " << clu5x9.towerHits[j].towerIet; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 number of towers " << clu5x9.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;
    }

    // Create and Fill the collection of 3D clusters and their usefull attributes
    std::unique_ptr<HGClusterHelper::HGClustersCollection> HGClustersCollection(new HGClusterHelper::HGClustersCollection);

    iEvent.getByToken(HGClusterToken, HGClusterHandle);
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
    }

    iEvent.put(std::move(l1TowerClusters5x9), "l1TowerClusters5x9");
    iEvent.put(std::move(HGClustersCollection), "HGClustersCollection");
}

int L1CaloTauProducer::tower_diPhi(int &iPhi_1, int &iPhi_2) const
{
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    if (result > PI)   { result -= 2 * PI; }
    if (result <= -PI) { result += 2 * PI; } 
    return result;
}

int L1CaloTauProducer::tower_diEta(int &iEta_1, int &iEta_2) const
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
    std::sort(begin(towerHits), end(towerHits), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b)
    { 
        if (a.towerIeta == b.towerIeta)
        {
            // if      ((a.towerIphi>=65 && a.towerIphi<=72) && (b.towerIphi>=1 && b.towerIphi<=8)) { return a.towerIphi > b.towerIphi; }
            // else if ((b.towerIphi>=65 && b.towerIphi<=72) && (a.towerIphi>=1 && a.towerIphi<=8)) { return a.towerIphi > b.towerIphi; }
            // else                                                                                 { return a.towerIphi < b.towerIphi; }

            if (((a.towerIphi>=65 && a.towerIphi<=72) && (b.towerIphi>=1 && b.towerIphi<=8)) ||
                ((b.towerIphi>=65 && b.towerIphi<=72) && (a.towerIphi>=1 && a.towerIphi<=8)))   { return a.towerIphi > b.towerIphi; }
            
            else { return a.towerIphi < b.towerIphi; }
        }
        else { return a.towerIeta < b.towerIeta; }
    });

    return towerHits;
}

DEFINE_FWK_MODULE(L1CaloTauProducer);