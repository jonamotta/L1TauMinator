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
        const std::vector<float> fixedHGCalTowerEtas = {1.52125, 1.60575, 1.69025, 1.77475, 1.85925, 1.94375, 2.02825, 2.11275, 2.19725, 2.28175, 2.36625, 2.45075, 2.53525, 2.61975, 2.70425, 2.78875, 2.87325, 2.95775};
        const std::vector<float> fixedHGCalTowerPhis = {0.0436332 , 0.1309 , 0.218166 , 0.305433 , 0.392699 , 0.479966 , 0.567232 , 0.654498 , 0.741765 , 0.829031 , 0.916298 ,
                                                        1.00356 , 1.09083 , 1.1781 , 1.26536 , 1.35263 , 1.4399 , 1.52716 , 1.61443 , 1.7017 , 1.78896 , 1.87623 , 1.9635 , 
                                                        2.05076 , 2.13803 , 2.22529 , 2.31256 , 2.39983 , 2.48709 , 2.57436 , 2.66163 , 2.74889 , 2.83616 , 2.92343 , 3.01069 , 3.09796, 
                                                        -3.09796 , -3.01069 , -2.92343 , -2.83616 , -2.74889 , -2.66163 , -2.57436 , -2.48709 , -2.39983 , -2.31256 , -2.22529 , -2.13803 , -2.05076 , 
                                                        -1.9635 , -1.87623 , -1.78896 , -1.7017 , -1.61443 , -1.52716 , -1.4399 , -1.35263 , -1.26536 , -1.1781 , -1.09083 , -1.00356 , 
                                                        -0.916298 , -0.829031 , -0.741765 , -0.654498 , -0.567232 , -0.479966 , -0.392699 , -0.305433 , -0.218166 , -0.1309 , -0.0436332};
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
        l1Hit.towerIeta    = ( find(fixedHGCalTowerEtas.begin(), fixedHGCalTowerEtas.end(), std::abs(l1Hit.towerEta)) - fixedHGCalTowerEtas.begin() + 17 + 1 ) * std::copysign(1,l1Hit.towerEta);
        l1Hit.towerIphi    = find(fixedHGCalTowerPhis.begin(), fixedHGCalTowerPhis.end(), std::abs(l1Hit.towerEta)) - fixedHGCalTowerPhis.begin() + 1;
        l1Hit.towerIem     = floor( l1Hit.towerEm/0.5 );
        l1Hit.towerIhad    = floor( l1Hit.towerHad/0.5 );
        l1Hit.towerIet     = floor( (l1Hit.towerEm + l1Hit.towerHad)/0.5 );

        l1CaloTowers.push_back(l1Hit);
    }

    // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
    std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b) {
      return a.towerEt > b.towerEt;
    });

    /********************************************************************************************
    * Begin with making CaloCandidates in 9x9 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters9x9(new TowerHelper::TowerClustersCollection);
    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters7x7(new TowerHelper::TowerClustersCollection);
    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters5x5(new TowerHelper::TowerClustersCollection);
    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters5x9(new TowerHelper::TowerClustersCollection);

    // Count the number of unused HCAL TPs so we can stop while loop after done.
    // Clustering can also stop once there are no seed hits >= EtMinForSeeding
    int n_towers = l1CaloTowers.size();
    int n_stale = 0;
    bool caloJetClusteringFinished = false;
    while (!caloJetClusteringFinished && n_towers != n_stale)
    {
        // 9x9 to 5x5 dimension decrease is the largest range of dimensions we can use to have always the same
        // seed fro all the masks and do not incur in prioblems with overlap or candidates multiplicity
        // incidentally this is also the dimensions range that makes sense to probe given the tau/jet footprints
        TowerHelper::TowerCluster clu9x9; clu9x9.Init();
        TowerHelper::TowerCluster clu7x7; clu7x7.Init();
        TowerHelper::TowerCluster clu5x5; clu5x5.Init();
        TowerHelper::TowerCluster clu5x9; clu5x9.Init();

        int cnt = 0;
        for (auto &l1CaloTower : l1CaloTowers)
        {
            cnt++;
            if (l1CaloTower.stale) { continue; }  // skip l1CaloTowers which are already used

            // find highest ET tower and use to seed the Candidates
            if (clu5x5.nHits == 0.0)  // this is the first l1CaloTower to seed the candidates
            {
                // the leading unused tower has ET < min, stop jet clustering
                if (l1CaloTower.towerEt < EtMinForSeeding)
                {
                    caloJetClusteringFinished = true;
                    continue;
                }
                l1CaloTower.stale = true;
                n_stale++;

                // Set seed location needed for delta iEta/iPhi
                if (l1CaloTower.isBarrel)
                { 
                    clu9x9.barrelSeeded = true;
                    clu7x7.barrelSeeded = true;
                    clu5x5.barrelSeeded = true;
                    clu5x9.barrelSeeded = true;
                }
                
                clu9x9.seedIeta = l1CaloTower.towerIeta;
                clu9x9.seedIphi = l1CaloTower.towerIphi;
                clu9x9.seedEta  = l1CaloTower.towerEta;
                clu9x9.seedPhi  = l1CaloTower.towerPhi;
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
                if      (clu9x9.seedIeta<=13) { clu9x9.isBarrel = true;  }
                else if (clu9x9.seedIeta>=22) { clu9x9.isEndcap = true;  }
                else                          { clu9x9.isOverlap = true; }
                clu9x9.nHits++;

                clu7x7.seedIeta = l1CaloTower.towerIeta;
                clu7x7.seedIphi = l1CaloTower.towerIphi;
                clu7x7.seedEta  = l1CaloTower.towerEta;
                clu7x7.seedPhi  = l1CaloTower.towerPhi;
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
                if      (clu7x7.seedIeta<=14) { clu7x7.isBarrel = true;  }
                else if (clu7x7.seedIeta>=21) { clu7x7.isEndcap = true;  }
                else                          { clu7x7.isOverlap = true; }
                clu7x7.nHits++;

                clu5x5.seedIeta = l1CaloTower.towerIeta;
                clu5x5.seedIphi = l1CaloTower.towerIphi;
                clu5x5.seedEta  = l1CaloTower.towerEta;
                clu5x5.seedPhi  = l1CaloTower.towerPhi;
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
                if      (clu5x5.seedIeta<=15) { clu5x5.isBarrel = true;  }
                else if (clu5x5.seedIeta>=20) { clu5x5.isEndcap = true;  }
                else                          { clu5x5.isOverlap = true; }
                clu5x5.nHits++;

                clu5x9.seedIeta = l1CaloTower.towerIeta;
                clu5x9.seedIphi = l1CaloTower.towerIphi;
                clu5x9.seedEta  = l1CaloTower.towerEta;
                clu5x9.seedPhi  = l1CaloTower.towerPhi;
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
                if      (clu5x9.seedIeta<=15) { clu5x9.isBarrel = true;  }
                else if (clu5x9.seedIeta>=20) { clu5x9.isEndcap = true;  }
                else                          { clu5x9.isOverlap = true; }
                clu5x9.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            // the defaults of 99 will automatically fail comparisons for the incorrect regions.
            int d_iEta = 99;
            int d_iPhi = 99;
            d_iEta = tower_diEta(clu5x5.seedIeta, l1CaloTower.towerIeta);
            d_iPhi = tower_diPhi(clu5x5.seedIphi, l1CaloTower.towerIphi);

            // 9x9 candidate mask
            if (abs(d_iEta) <= 4 && abs(d_iPhi) <= 4)
            {
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
                clu9x9.nHits++;
            }

            // 7x7 candidate mask
            if (abs(d_iEta) <= 3 && abs(d_iPhi) <= 3)
            {
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
                clu7x7.nHits++;
            }

            // 5x5 candidate mask
            if (abs(d_iEta) <= 2 && abs(d_iPhi) <= 2)
            {
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
                clu5x5.nHits++;
            }

            // 5x9 candidate mask
            if (abs(d_iEta) <= 2 && abs(d_iPhi) <= 4)
            {
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
                clu5x9.nHits++;
            }
        }
            
        if (clu5x5.nHits > 0.0) { 
            l1TowerClusters9x9->push_back(clu9x9);
            l1TowerClusters7x7->push_back(clu7x7);
            l1TowerClusters5x5->push_back(clu5x5);
            l1TowerClusters5x9->push_back(clu5x9);
        }

    }  // end while loop of HCAL TP clustering

    if (DEBUG)
    {
        for (auto& clu9x9 : *l1TowerClusters9x9)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 9x9 seed " << " , eta " << clu9x9.seedIeta << " phi " << clu9x9.seedIphi << std::endl;
            std::cout << " -- clu 9x9 seed " << " , isBarrel " << clu9x9.isBarrel << " isEndcap " << clu9x9.isEndcap << " isOverlap " << clu9x9.isOverlap << std::endl;
            std::cout << " -- clu 9x9 towers etas [" << clu9x9.seedIeta;
            for (long unsigned int j = 0; j < clu9x9.towerIeta.size(); ++j) { std::cout << clu9x9.towerIeta[j]; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers phis [" << clu9x9.seedIphi;
            for (long unsigned int j = 0; j < clu9x9.towerIphi.size(); ++j) { std::cout << ", " << clu9x9.towerIphi[j]; }
            std::cout << "]" << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;
    }

    iEvent.put(std::move(l1TowerClusters9x9), "l1TowerClusters9x9");
    iEvent.put(std::move(l1TowerClusters7x7), "l1TowerClusters7x7");
    iEvent.put(std::move(l1TowerClusters5x5), "l1TowerClusters5x5");
    iEvent.put(std::move(l1TowerClusters5x9), "l1TowerClusters5x9");
}

int CaloTowerHandler::tower_diPhi(int &iPhi_1, int &iPhi_2) const {
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    while (result > PI)
        result -= 2 * PI;
    while (result <= -PI)
        result += 2 * PI;
    return result;
}

int CaloTowerHandler::tower_diEta(int &iEta_1, int &iEta_2) const {
    if (iEta_1 * iEta_2 > 0)
        return iEta_1 - iEta_2;
    else
        return iEta_1 - iEta_2 - 1;
}

DEFINE_FWK_MODULE(CaloTowerHandler);