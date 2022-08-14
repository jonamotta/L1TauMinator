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

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
// #include "L1TauMinator/L1TauMinatorEmulator/interface/TowerHelper.h"

class CaloTowerHandler : public edm::stream::EDProducer<> {
    public:
        explicit CaloTowerHandler(const edm::ParameterSet&);

    private:
        //----edm control---
        void produce(edm::Event&, const edm::EventSetup&) override;

        //----private functions----
        int tower_diPhi(int &iPhi_1, int &iPhi_2) const;
        int tower_diEta(int &iEta_1, int &iEta_2) const;
        int endcap_iphi(float &phi) const;
        int endcap_ieta(float &eta) const;

        //----tokens and handles----
        edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowerToken;
        edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

        edm::EDGetToken hgcalTowersToken;
        edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;

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
CaloTowerHandler::CaloTowerHandler(const edm::ParameterSet& iConfig) 
    : l1TowerToken(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
      hgcalTowersToken(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalTowers"))),
      EcalEtMinForClustering(iConfig.getParameter<double>("EcalEtMinForClustering")),
      HcalEtMinForClustering(iConfig.getParameter<double>("HcalEtMinForClustering")),
      EtMinForSeeding(iConfig.getParameter<double>("EtMinForSeeding")),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters9x9");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters7x7");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters5x5");
    produces<TowerHelper::TowerClustersCollection>("l1TowerClusters5x9");

    if (DEBUG) { std::cout << "EtMinForSeeding = " << EtMinForSeeding << " , HcalTpEtMin = " << HcalEtMinForClustering << " , EcalTpEtMin = " << EcalEtMinForClustering << std::endl; }
}

void CaloTowerHandler::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill collection of all calotowers and their attributes
    std::vector<TowerHelper::TowerHit> l1CaloTowers;

    iEvent.getByToken(l1TowerToken, l1CaloTowerHandle);
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

    /********************************************************************************************
    * Begin with making TowerClusters in 9x9 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters9x9(new TowerHelper::TowerClustersCollection);

    // initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 9x9 TowerClusters seeds finding
    bool caloJetSeedingFinished = false;
    while (!caloJetSeedingFinished)
    {
        TowerHelper::TowerCluster clu9x9; clu9x9.Init();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu9x9.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu9x9.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu9x9.seedIeta = l1CaloTower.towerIeta;
                clu9x9.seedIphi = l1CaloTower.towerIphi;
                clu9x9.seedEta  = l1CaloTower.towerEta;
                clu9x9.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu9x9.seedIeta)<=13) { clu9x9.isBarrel = true;  }
                else if (abs(clu9x9.seedIeta)>=22) { clu9x9.isEndcap = true;  }
                else                               { clu9x9.isOverlap = true; }

                // Fill the TowerCluster towers variables
                clu9x9.towerEta.push_back(l1CaloTower.towerEta);
                clu9x9.towerPhi.push_back(l1CaloTower.towerPhi);
                clu9x9.towerEm.push_back(l1CaloTower.towerEm);
                clu9x9.towerHad.push_back(l1CaloTower.towerHad);
                clu9x9.towerEt.push_back(l1CaloTower.towerEt);
                clu9x9.towerIeta.push_back(l1CaloTower.towerIeta);
                clu9x9.towerIphi.push_back(l1CaloTower.towerIphi);
                clu9x9.towerIem.push_back(l1CaloTower.towerIem);
                clu9x9.towerIhad.push_back(l1CaloTower.towerIhad);
                clu9x9.towerIet.push_back(l1CaloTower.towerIet);
                
                // Fill the TowerCluster overall variables
                clu9x9.totalEm += l1CaloTower.towerEm;
                clu9x9.totalHad += l1CaloTower.towerHad;
                clu9x9.totalEt += l1CaloTower.towerEt;
                clu9x9.totalIem += l1CaloTower.towerIem;
                clu9x9.totalIhad += l1CaloTower.towerIhad;
                clu9x9.totalIet += l1CaloTower.towerIet;
                clu9x9.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu9x9.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu9x9.seedIphi, l1CaloTower.towerIphi);

            // stale tower for seeding if it would lead to overalp between clusters
            if (abs(d_iEta) <= 8 && abs(d_iPhi) <= 8) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu9x9.nHits > 0.0) { l1TowerClusters9x9->push_back(clu9x9); }

    }  // end while loop of TowerClusters seeding

    // loop for 9x9 TowerClusters creation starting from the seed just found
    for (auto& clu9x9 : *l1TowerClusters9x9)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu9x9.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu9x9.seedIphi, l1CaloTower.towerIphi);

            // cluster all towers in a 9x9 towers mask
            if (abs(d_iEta) <= 4 && abs(d_iPhi) <= 4)
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers variables
                clu9x9.towerEta.push_back(l1CaloTower.towerEta);
                clu9x9.towerPhi.push_back(l1CaloTower.towerPhi);
                clu9x9.towerEm.push_back(l1CaloTower.towerEm);
                clu9x9.towerHad.push_back(l1CaloTower.towerHad);
                clu9x9.towerEt.push_back(l1CaloTower.towerEt);
                clu9x9.towerIeta.push_back(l1CaloTower.towerIeta);
                clu9x9.towerIphi.push_back(l1CaloTower.towerIphi);
                clu9x9.towerIem.push_back(l1CaloTower.towerIem);
                clu9x9.towerIhad.push_back(l1CaloTower.towerIhad);
                clu9x9.towerIet.push_back(l1CaloTower.towerIet);

                // Fill the TowerCluster overall variables
                clu9x9.totalEm += l1CaloTower.towerEm;
                clu9x9.totalHad += l1CaloTower.towerHad;
                clu9x9.totalEt += l1CaloTower.towerEt;
                clu9x9.totalIem += l1CaloTower.towerIem;
                clu9x9.totalIhad += l1CaloTower.towerIhad;
                clu9x9.totalIet += l1CaloTower.towerIet;
                if (l1CaloTower.towerIet > 0) clu9x9.nHits++;
            }
        }// end for loop of TP clustering
    }// end while loop of 9x9 TowerClusters creation


    /********************************************************************************************
    * Begin with making TowerClusters in 7x7 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters7x7(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 7x7 TowerClusters seeds finding
    caloJetSeedingFinished = false;
    while (!caloJetSeedingFinished)
    {
        TowerHelper::TowerCluster clu7x7; clu7x7.Init();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu7x7.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu7x7.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu7x7.seedIeta = l1CaloTower.towerIeta;
                clu7x7.seedIphi = l1CaloTower.towerIphi;
                clu7x7.seedEta  = l1CaloTower.towerEta;
                clu7x7.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu7x7.seedIeta)<=13) { clu7x7.isBarrel = true;  }
                else if (abs(clu7x7.seedIeta)>=22) { clu7x7.isEndcap = true;  }
                else                               { clu7x7.isOverlap = true; }

                // Fill the TowerCluster towers variables
                clu7x7.towerEta.push_back(l1CaloTower.towerEta);
                clu7x7.towerPhi.push_back(l1CaloTower.towerPhi);
                clu7x7.towerEm.push_back(l1CaloTower.towerEm);
                clu7x7.towerHad.push_back(l1CaloTower.towerHad);
                clu7x7.towerEt.push_back(l1CaloTower.towerEt);
                clu7x7.towerIeta.push_back(l1CaloTower.towerIeta);
                clu7x7.towerIphi.push_back(l1CaloTower.towerIphi);
                clu7x7.towerIem.push_back(l1CaloTower.towerIem);
                clu7x7.towerIhad.push_back(l1CaloTower.towerIhad);
                clu7x7.towerIet.push_back(l1CaloTower.towerIet);
                
                // Fill the TowerCluster overall variables
                clu7x7.totalEm += l1CaloTower.towerEm;
                clu7x7.totalHad += l1CaloTower.towerHad;
                clu7x7.totalEt += l1CaloTower.towerEt;
                clu7x7.totalIem += l1CaloTower.towerIem;
                clu7x7.totalIhad += l1CaloTower.towerIhad;
                clu7x7.totalIet += l1CaloTower.towerIet;
                clu7x7.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu7x7.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu7x7.seedIphi, l1CaloTower.towerIphi);

            // stale tower for seeding if it would lead to overalp between clusters
            if (abs(d_iEta) <= 6 && abs(d_iPhi) <= 6) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu7x7.nHits > 0.0) { l1TowerClusters7x7->push_back(clu7x7); }

    }  // end while loop of TowerClusters seeding

    // loop for 7x7 TowerClusters creation starting from the seed just found
    for (auto& clu7x7 : *l1TowerClusters7x7)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu7x7.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu7x7.seedIphi, l1CaloTower.towerIphi);

            // cluster all towers in a 7x7 towers mask
            if (abs(d_iEta) <= 3 && abs(d_iPhi) <= 3)
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers variables
                clu7x7.towerEta.push_back(l1CaloTower.towerEta);
                clu7x7.towerPhi.push_back(l1CaloTower.towerPhi);
                clu7x7.towerEm.push_back(l1CaloTower.towerEm);
                clu7x7.towerHad.push_back(l1CaloTower.towerHad);
                clu7x7.towerEt.push_back(l1CaloTower.towerEt);
                clu7x7.towerIeta.push_back(l1CaloTower.towerIeta);
                clu7x7.towerIphi.push_back(l1CaloTower.towerIphi);
                clu7x7.towerIem.push_back(l1CaloTower.towerIem);
                clu7x7.towerIhad.push_back(l1CaloTower.towerIhad);
                clu7x7.towerIet.push_back(l1CaloTower.towerIet);

                // Fill the TowerCluster overall variables
                clu7x7.totalEm += l1CaloTower.towerEm;
                clu7x7.totalHad += l1CaloTower.towerHad;
                clu7x7.totalEt += l1CaloTower.towerEt;
                clu7x7.totalIem += l1CaloTower.towerIem;
                clu7x7.totalIhad += l1CaloTower.towerIhad;
                clu7x7.totalIet += l1CaloTower.towerIet;
                if (l1CaloTower.towerIet > 0) clu7x7.nHits++;
            }
        }// end for loop of TP clustering
    }// end while loop of 7x7 TowerClusters creation


    /********************************************************************************************
    * Begin with making TowerClusters in 5x5 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters5x5(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 5x5 TowerClusters seeds finding
    caloJetSeedingFinished = false;
    while (!caloJetSeedingFinished)
    {
        TowerHelper::TowerCluster clu5x5; clu5x5.Init();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu5x5.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu5x5.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu5x5.seedIeta = l1CaloTower.towerIeta;
                clu5x5.seedIphi = l1CaloTower.towerIphi;
                clu5x5.seedEta  = l1CaloTower.towerEta;
                clu5x5.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu5x5.seedIeta)<=13) { clu5x5.isBarrel = true;  }
                else if (abs(clu5x5.seedIeta)>=22) { clu5x5.isEndcap = true;  }
                else                               { clu5x5.isOverlap = true; }

                // Fill the TowerCluster towers variables
                clu5x5.towerEta.push_back(l1CaloTower.towerEta);
                clu5x5.towerPhi.push_back(l1CaloTower.towerPhi);
                clu5x5.towerEm.push_back(l1CaloTower.towerEm);
                clu5x5.towerHad.push_back(l1CaloTower.towerHad);
                clu5x5.towerEt.push_back(l1CaloTower.towerEt);
                clu5x5.towerIeta.push_back(l1CaloTower.towerIeta);
                clu5x5.towerIphi.push_back(l1CaloTower.towerIphi);
                clu5x5.towerIem.push_back(l1CaloTower.towerIem);
                clu5x5.towerIhad.push_back(l1CaloTower.towerIhad);
                clu5x5.towerIet.push_back(l1CaloTower.towerIet);
                
                // Fill the TowerCluster overall variables
                clu5x5.totalEm += l1CaloTower.towerEm;
                clu5x5.totalHad += l1CaloTower.towerHad;
                clu5x5.totalEt += l1CaloTower.towerEt;
                clu5x5.totalIem += l1CaloTower.towerIem;
                clu5x5.totalIhad += l1CaloTower.towerIhad;
                clu5x5.totalIet += l1CaloTower.towerIet;
                clu5x5.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu5x5.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu5x5.seedIphi, l1CaloTower.towerIphi);

            // stale tower for seeding if it would lead to overalp between clusters
            if (abs(d_iEta) <= 4 && abs(d_iPhi) <= 4) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu5x5.nHits > 0.0) { l1TowerClusters5x5->push_back(clu5x5); }

    }  // end while loop of TowerClusters seeding

    // loop for 5x5 TowerClusters creation starting from the seed just found
    for (auto& clu5x5 : *l1TowerClusters5x5)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int d_iEta = tower_diEta(clu5x5.seedIeta, l1CaloTower.towerIeta);
            int d_iPhi = tower_diPhi(clu5x5.seedIphi, l1CaloTower.towerIphi);

            // cluster all towers in a 5x5 towers mask
            if (abs(d_iEta) <= 2 && abs(d_iPhi) <= 2)
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers variables
                clu5x5.towerEta.push_back(l1CaloTower.towerEta);
                clu5x5.towerPhi.push_back(l1CaloTower.towerPhi);
                clu5x5.towerEm.push_back(l1CaloTower.towerEm);
                clu5x5.towerHad.push_back(l1CaloTower.towerHad);
                clu5x5.towerEt.push_back(l1CaloTower.towerEt);
                clu5x5.towerIeta.push_back(l1CaloTower.towerIeta);
                clu5x5.towerIphi.push_back(l1CaloTower.towerIphi);
                clu5x5.towerIem.push_back(l1CaloTower.towerIem);
                clu5x5.towerIhad.push_back(l1CaloTower.towerIhad);
                clu5x5.towerIet.push_back(l1CaloTower.towerIet);

                // Fill the TowerCluster overall variables
                clu5x5.totalEm += l1CaloTower.towerEm;
                clu5x5.totalHad += l1CaloTower.towerHad;
                clu5x5.totalEt += l1CaloTower.towerEt;
                clu5x5.totalIem += l1CaloTower.towerIem;
                clu5x5.totalIhad += l1CaloTower.towerIhad;
                clu5x5.totalIet += l1CaloTower.towerIet;
                if (l1CaloTower.towerIet > 0) clu5x5.nHits++;
            }
        }// end for loop of TP clustering
    }// end while loop of 5x5 TowerClusters creation


    /********************************************************************************************
    * Begin with making TowerClusters in 5x9 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters5x9(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 5x9 TowerClusters seeds finding
    caloJetSeedingFinished = false;
    while (!caloJetSeedingFinished)
    {
        TowerHelper::TowerCluster clu5x9; clu5x9.Init();

        for (auto &l1CaloTower : l1CaloTowers)
        {
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

                // Fill the TowerCluster towers variables
                clu5x9.towerEta.push_back(l1CaloTower.towerEta);
                clu5x9.towerPhi.push_back(l1CaloTower.towerPhi);
                clu5x9.towerEm.push_back(l1CaloTower.towerEm);
                clu5x9.towerHad.push_back(l1CaloTower.towerHad);
                clu5x9.towerEt.push_back(l1CaloTower.towerEt);
                clu5x9.towerIeta.push_back(l1CaloTower.towerIeta);
                clu5x9.towerIphi.push_back(l1CaloTower.towerIphi);
                clu5x9.towerIem.push_back(l1CaloTower.towerIem);
                clu5x9.towerIhad.push_back(l1CaloTower.towerIhad);
                clu5x9.towerIet.push_back(l1CaloTower.towerIet);
                
                // Fill the TowerCluster overall variables
                clu5x9.totalEm += l1CaloTower.towerEm;
                clu5x9.totalHad += l1CaloTower.towerHad;
                clu5x9.totalEt += l1CaloTower.towerEt;
                clu5x9.totalIem += l1CaloTower.towerIem;
                clu5x9.totalIhad += l1CaloTower.towerIhad;
                clu5x9.totalIet += l1CaloTower.towerIet;
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

                // Fill the TowerCluster towers variables
                clu5x9.towerEta.push_back(l1CaloTower.towerEta);
                clu5x9.towerPhi.push_back(l1CaloTower.towerPhi);
                clu5x9.towerEm.push_back(l1CaloTower.towerEm);
                clu5x9.towerHad.push_back(l1CaloTower.towerHad);
                clu5x9.towerEt.push_back(l1CaloTower.towerEt);
                clu5x9.towerIeta.push_back(l1CaloTower.towerIeta);
                clu5x9.towerIphi.push_back(l1CaloTower.towerIphi);
                clu5x9.towerIem.push_back(l1CaloTower.towerIem);
                clu5x9.towerIhad.push_back(l1CaloTower.towerIhad);
                clu5x9.towerIet.push_back(l1CaloTower.towerIet);

                // Fill the TowerCluster overall variables
                clu5x9.totalEm += l1CaloTower.towerEm;
                clu5x9.totalHad += l1CaloTower.towerHad;
                clu5x9.totalEt += l1CaloTower.towerEt;
                clu5x9.totalIem += l1CaloTower.towerIem;
                clu5x9.totalIhad += l1CaloTower.towerIhad;
                clu5x9.totalIet += l1CaloTower.towerIet;
                if (l1CaloTower.towerIet > 0) clu5x9.nHits++;
            }
        }// end for loop of TP clustering
    }// end while loop of 5x9 TowerClusters creation


    if (DEBUG) 
    {
        std::cout << "\n***************************************************************************************************************************************" << std::endl;
        std::cout << "***************************************************************************************************************************************\n" << std::endl;

        for (auto& clu9x9 : *l1TowerClusters9x9)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 9x9 seed " << " , eta " << clu9x9.seedIeta << " phi " << clu9x9.seedIphi << std::endl;
            std::cout << " -- clu 9x9 seed " << " , isBarrel " << clu9x9.isBarrel << " isEndcap " << clu9x9.isEndcap << " isOverlap " << clu9x9.isOverlap << std::endl;
            std::cout << " -- clu 9x9 towers etas (" << clu9x9.towerIeta.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerIeta.size(); ++j) { std::cout  << ", " << clu9x9.towerIeta[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers phis (" << clu9x9.towerIphi.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerIphi.size(); ++j) { std::cout << ", " << clu9x9.towerIphi[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers ems (" << clu9x9.towerIem.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerIem.size(); ++j) { std::cout << ", " << clu9x9.towerIem[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers hads (" << clu9x9.towerIhad.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerIhad.size(); ++j) { std::cout << ", " << clu9x9.towerIhad[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers ets (" << clu9x9.towerIet.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerIet.size(); ++j) { std::cout << ", " << clu9x9.towerIet[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 number of towers " << clu9x9.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;

        for (auto& clu7x7 : *l1TowerClusters7x7)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 7x7 seed " << " , eta " << clu7x7.seedIeta << " phi " << clu7x7.seedIphi << std::endl;
            std::cout << " -- clu 7x7 seed " << " , isBarrel " << clu7x7.isBarrel << " isEndcap " << clu7x7.isEndcap << " isOverlap " << clu7x7.isOverlap << std::endl;
            std::cout << " -- clu 7x7 towers etas (" << clu7x7.towerIeta.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerIeta.size(); ++j) { std::cout  << ", " << clu7x7.towerIeta[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers phis (" << clu7x7.towerIphi.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerIphi.size(); ++j) { std::cout << ", " << clu7x7.towerIphi[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers ems (" << clu7x7.towerIem.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerIem.size(); ++j) { std::cout << ", " << clu7x7.towerIem[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers hads (" << clu7x7.towerIhad.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerIhad.size(); ++j) { std::cout << ", " << clu7x7.towerIhad[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers ets (" << clu7x7.towerIet.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerIet.size(); ++j) { std::cout << ", " << clu7x7.towerIet[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 number of towers " << clu7x7.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;

        for (auto& clu5x5 : *l1TowerClusters5x5)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 5x5 seed " << " , eta " << clu5x5.seedIeta << " phi " << clu5x5.seedIphi << std::endl;
            std::cout << " -- clu 5x5 seed " << " , isBarrel " << clu5x5.isBarrel << " isEndcap " << clu5x5.isEndcap << " isOverlap " << clu5x5.isOverlap << std::endl;
            std::cout << " -- clu 5x5 towers etas (" << clu5x5.towerIeta.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerIeta.size(); ++j) { std::cout  << ", " << clu5x5.towerIeta[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers phis (" << clu5x5.towerIphi.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerIphi.size(); ++j) { std::cout << ", " << clu5x5.towerIphi[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers ems (" << clu5x5.towerIem.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerIem.size(); ++j) { std::cout << ", " << clu5x5.towerIem[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers hads (" << clu5x5.towerIhad.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerIhad.size(); ++j) { std::cout << ", " << clu5x5.towerIhad[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers ets (" << clu5x5.towerIet.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerIet.size(); ++j) { std::cout << ", " << clu5x5.towerIet[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 number of towers " << clu5x5.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;

        for (auto& clu5x9 : *l1TowerClusters5x9)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 5x9 seed " << " , eta " << clu5x9.seedIeta << " phi " << clu5x9.seedIphi << std::endl;
            std::cout << " -- clu 5x9 seed " << " , isBarrel " << clu5x9.isBarrel << " isEndcap " << clu5x9.isEndcap << " isOverlap " << clu5x9.isOverlap << std::endl;
            std::cout << " -- clu 5x9 towers etas (" << clu5x9.towerIeta.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerIeta.size(); ++j) { std::cout  << ", " << clu5x9.towerIeta[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers phis (" << clu5x9.towerIphi.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerIphi.size(); ++j) { std::cout << ", " << clu5x9.towerIphi[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers ems (" << clu5x9.towerIem.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerIem.size(); ++j) { std::cout << ", " << clu5x9.towerIem[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers hads (" << clu5x9.towerIhad.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerIhad.size(); ++j) { std::cout << ", " << clu5x9.towerIhad[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers ets (" << clu5x9.towerIet.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerIet.size(); ++j) { std::cout << ", " << clu5x9.towerIet[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 number of towers " << clu5x9.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;
    }

    iEvent.put(std::move(l1TowerClusters9x9), "l1TowerClusters9x9");
    iEvent.put(std::move(l1TowerClusters7x7), "l1TowerClusters7x7");
    iEvent.put(std::move(l1TowerClusters5x5), "l1TowerClusters5x5");
    iEvent.put(std::move(l1TowerClusters5x9), "l1TowerClusters5x9");
}

int CaloTowerHandler::tower_diPhi(int &iPhi_1, int &iPhi_2) const
{
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    if (result > PI)   { result -= 2 * PI; }
    if (result <= -PI) { result += 2 * PI; } 
    return result;
}

int CaloTowerHandler::tower_diEta(int &iEta_1, int &iEta_2) const
{
    if (iEta_1 * iEta_2 > 0) { return iEta_1 - iEta_2; }
    else
    {
        if (iEta_1>0) { return iEta_1 - iEta_2 - 1; }
        else          { return iEta_1 - iEta_2 + 1; }
    }
}

int CaloTowerHandler::endcap_iphi(float &phi) const
{
    float phi_step = 0.0872664;
    if (phi > 0) { return floor(phi / phi_step) + 1;  }
    else         { return floor(phi / phi_step) + 73; }
}

int CaloTowerHandler::endcap_ieta(float &eta) const
{
    float eta_step = 0.0845;
    return floor(abs(eta)/eta_step) * std::copysign(1,eta);
}

DEFINE_FWK_MODULE(CaloTowerHandler);