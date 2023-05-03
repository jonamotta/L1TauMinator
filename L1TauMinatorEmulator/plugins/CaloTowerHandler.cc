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
        std::vector<TowerHelper::TowerHit> sortPicLikeI(std::vector<TowerHelper::TowerHit>) const;
        std::vector<TowerHelper::TowerHit> sortPicLikeF(std::vector<TowerHelper::TowerHit>) const;

        //----tokens and handles----
        edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowersToken;
        edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

        edm::EDGetToken hgcalTowersToken;
        edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;

        edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalDigisToken;
        edm::Handle<HcalTrigPrimDigiCollection> hcalDigisHandle;
        edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderTag;

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
    : l1TowersToken(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
      hgcalTowersToken(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("hgcalTowers"))),
      hcalDigisToken(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigis"))),
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

    std::cout << "EtMinForSeeding = " << EtMinForSeeding << " , HcalTpEtMin = " << HcalEtMinForClustering << " , EcalTpEtMin = " << EcalEtMinForClustering << std::endl;
}

void CaloTowerHandler::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill collection of all calotowers and their attributes
    std::vector<TowerHelper::TowerHit> l1CaloTowers;

    iEvent.getByToken(l1TowersToken, l1CaloTowerHandle);
    int warnings = 0;
    for (auto &hit : *l1CaloTowerHandle.product())
    {
        // skip this weird towers and store warning
        if (hit.towerIEta() == -1016 && hit.towerIPhi() == -962)
        {
            warnings += 1;
            continue;
        }

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

    // Fill missing towers in HGCal due to current simulation limitation (because right now they are produced starting from modules
    // and some TT do not have a centered module in front of them) --> just fill with zeros (JB approves)
    // At the same time this will fill the missing towers from IEta=-1016 IPhi=-962
    std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b)
    {
        if (a.towerIeta == b.towerIeta) { return a.towerIphi < b.towerIphi; }
        else                            { return a.towerIeta < b.towerIeta; }
    });
    int iEtaProgression = -35;
    int iPhiProgression = 1;
    for (auto &l1CaloTower : l1CaloTowers)
    {
        // skip towers with the weird IEta=-1016 IPhi=-962 values and they will also be filled with zeros
        if (l1CaloTower.towerIeta < -35) { continue; }

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
                // 0.043650 == first tower eta
                // 0.08730 == eta step in the barrel
                // 0.08080 == eta step between barrel and endcap
                // 0.08450 == eta step in the endcap
                int absEta = abs(iEtaProgression);
                int sgnEta = std::copysign(1,iEtaProgression);
                if (absEta<=17) { l1Hit.towerEta = sgnEta * (0.043650 + (absEta-1) * 0.08730); }
                else            { l1Hit.towerEta = sgnEta * (0.043650 + 16 * 0.08730 + 0.08080 + (absEta-18) * 0.08450); }
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
                    // 0.043650 == first tower eta
                    // 0.08730 == eta step in the barrel
                    // 0.08080 == eta step between barrel and endcap
                    // 0.08450 == eta step in the endcap
                    int absEta = abs(iEtaProgression);
                    int sgnEta = std::copysign(1,iEtaProgression);
                    if (absEta<=17) { l1Hit.towerEta = sgnEta * (0.043650 + (absEta-1) * 0.08730); }
                    else            { l1Hit.towerEta = sgnEta * (0.043650 + 16 * 0.08730 + 0.08080 + (absEta-18) * 0.08450); }
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

        if (iEtaProgression>35) { break; }
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

    /********************************************************************************************
    * Begin with making TowerClusters in 9x9 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters9x9(new TowerHelper::TowerClustersCollection);

    // initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 9x9 TowerClusters seeds finding
    bool caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu9x9; clu9x9.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            if (DEBUG) { std::cout << " // Ieta " << l1CaloTower.towerIeta << " - Iphi " << l1CaloTower.towerIphi; }

            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.65) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu9x9.nHits == 0.0)
            {
                // the leading unused tower has ET < min, stop jet clustering
                if (l1CaloTower.towerEt < EtMinForSeeding)
                {
                    caloTauSeedingFinished = true;
                    break;
                }
                l1CaloTower.stale4seed = true;
                l1CaloTower.stale = true;

                if (DEBUG) { std::cout << " SEED"; }

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
                clu9x9.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu9x9.totalEm      += l1CaloTower.towerEm;
                clu9x9.totalHad     += l1CaloTower.towerHad;
                clu9x9.totalEt      += l1CaloTower.towerEt;
                clu9x9.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu9x9.totalIem     += l1CaloTower.towerIem;
                clu9x9.totalIhad    += l1CaloTower.towerIhad;
                clu9x9.totalIet     += l1CaloTower.towerIet;
                clu9x9.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu9x9.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu9x9.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu9x9.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu9x9.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu9x9.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu9x9.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 8 && abs(d_iPhi) <= 8) || (abs(d_Eta) < 0.7 && abs(d_Phi) < 0.7)) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu9x9.nHits > 0.0) { l1TowerClusters9x9->push_back(clu9x9); }

        if (DEBUG) { std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl; }

    }  // end while loop of TowerClusters seeding

    if (DEBUG) { std::cout << "***************************************************************************************************************************************" << std::endl; }

    // loop for 9x9 TowerClusters creation starting from the seed just found
    for (auto& clu9x9 : *l1TowerClusters9x9)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            if (DEBUG) { std::cout << " // Ieta " << l1CaloTower.towerIeta << " - Iphi " << l1CaloTower.towerIphi; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu9x9.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu9x9.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu9x9.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu9x9.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu9x9.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 9x9 towers mask
            if ((abs(d_iEta) <= 4 && abs(d_iPhi) <= 4) || (abs(d_Eta) < 0.4 && abs(d_Phi) < 0.4))
            {
                if (DEBUG) { std::cout << " CLUSTERED"; }

                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu9x9.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu9x9.totalEm      += l1CaloTower.towerEm;
                clu9x9.totalHad     += l1CaloTower.towerHad;
                clu9x9.totalEt      += l1CaloTower.towerEt;
                clu9x9.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu9x9.totalIem     += l1CaloTower.towerIem;
                clu9x9.totalIhad    += l1CaloTower.towerIhad;
                clu9x9.totalIet     += l1CaloTower.towerIet;
                clu9x9.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu9x9.nHits++;
            }

        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu9x9.towerHits);
        clu9x9.InitHits(); clu9x9.towerHits = sortedHits;

        if (sortedHits.size() != 81) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 81 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

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
    caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu7x7; clu7x7.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.75) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu7x7.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu7x7.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu7x7.seedIeta = l1CaloTower.towerIeta;
                clu7x7.seedIphi = l1CaloTower.towerIphi;
                clu7x7.seedEta  = l1CaloTower.towerEta;
                clu7x7.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu7x7.seedIeta)<=13) { clu7x7.isBarrel = true;  }
                else if (abs(clu7x7.seedIeta)>=22) { clu7x7.isEndcap = true;  }
                else                               { clu7x7.isOverlap = true; }

                // Fill the TowerCluster towers
                clu7x7.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu7x7.totalEm      += l1CaloTower.towerEm;
                clu7x7.totalHad     += l1CaloTower.towerHad;
                clu7x7.totalEt      += l1CaloTower.towerEt;
                clu7x7.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu7x7.totalIem     += l1CaloTower.towerIem;
                clu7x7.totalIhad    += l1CaloTower.towerIhad;
                clu7x7.totalIet     += l1CaloTower.towerIet;
                clu7x7.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu7x7.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu7x7.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu7x7.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu7x7.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu7x7.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu7x7.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 6 && abs(d_iPhi) <= 6) || (abs(d_Eta) < 0.55 && abs(d_Phi) < 0.55)) { l1CaloTower.stale4seed = true; }
    
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
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu7x7.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu7x7.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu7x7.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu7x7.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu7x7.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 7x7 towers mask
            if ((abs(d_iEta) <= 3 && abs(d_iPhi) <= 3) || (abs(d_Eta) < 0.3 && abs(d_Phi) < 0.3))
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu7x7.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu7x7.totalEm      += l1CaloTower.towerEm;
                clu7x7.totalHad     += l1CaloTower.towerHad;
                clu7x7.totalEt      += l1CaloTower.towerEt;
                clu7x7.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu7x7.totalIem     += l1CaloTower.towerIem;
                clu7x7.totalIhad    += l1CaloTower.towerIhad;
                clu7x7.totalIet     += l1CaloTower.towerIet;
                clu7x7.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu7x7.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu7x7.towerHits);
        clu7x7.InitHits(); clu7x7.towerHits = sortedHits;

        if (sortedHits.size() != 49) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 49 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

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
    caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu5x5; clu5x5.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.83) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu5x5.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu5x5.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu5x5.seedIeta = l1CaloTower.towerIeta;
                clu5x5.seedIphi = l1CaloTower.towerIphi;
                clu5x5.seedEta  = l1CaloTower.towerEta;
                clu5x5.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu5x5.seedIeta)<=13) { clu5x5.isBarrel = true;  }
                else if (abs(clu5x5.seedIeta)>=22) { clu5x5.isEndcap = true;  }
                else                               { clu5x5.isOverlap = true; }

                // Fill the TowerCluster towers
                clu5x5.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu5x5.totalEm      += l1CaloTower.towerEm;
                clu5x5.totalHad     += l1CaloTower.towerHad;
                clu5x5.totalEt      += l1CaloTower.towerEt;
                clu5x5.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu5x5.totalIem     += l1CaloTower.towerIem;
                clu5x5.totalIhad    += l1CaloTower.towerIhad;
                clu5x5.totalIet     += l1CaloTower.towerIet;
                clu5x5.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu5x5.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu5x5.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu5x5.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu5x5.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu5x5.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu5x5.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 4 && abs(d_iPhi) <= 4) || (abs(d_Eta) < 0.35 && abs(d_Phi) < 0.35)) { l1CaloTower.stale4seed = true; }
    
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
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu5x5.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu5x5.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu5x5.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu5x5.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu5x5.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 5x5 towers mask
            if ((abs(d_iEta) <= 2 && abs(d_iPhi) <= 2) || (abs(d_Eta) < 0.2 && abs(d_Phi) < 0.2))
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu5x5.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu5x5.totalEm      += l1CaloTower.towerEm;
                clu5x5.totalHad     += l1CaloTower.towerHad;
                clu5x5.totalEt      += l1CaloTower.towerEt;
                clu5x5.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu5x5.totalIem     += l1CaloTower.towerIem;
                clu5x5.totalIhad    += l1CaloTower.towerIhad;
                clu5x5.totalIet     += l1CaloTower.towerIet;
                clu5x5.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu5x5.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu5x5.towerHits);
        clu5x5.InitHits(); clu5x5.towerHits = sortedHits;

        if (sortedHits.size() != 25) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 25 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

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
    caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu5x9; clu5x9.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.83) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu5x9.nHits == 0.0)
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
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu5x9.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu5x9.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu5x9.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu5x9.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu5x9.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 4 && abs(d_iPhi) <= 8) || (abs(d_Eta) < 0.35 && abs(d_Phi) < 0.7)) { l1CaloTower.stale4seed = true; }
    
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
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu5x9.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu5x9.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu5x9.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu5x9.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu5x9.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 5x9 towers mask
            if ((abs(d_iEta) <= 2 && abs(d_iPhi) <= 4) || (abs(d_Eta) < 0.2 && abs(d_Phi) < 0.4))
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
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu5x9.towerHits);
        clu5x9.InitHits(); clu5x9.towerHits = sortedHits;

        if (sortedHits.size() != 45) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 45 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

    }// end while loop of 5x9 TowerClusters creation

    /********************************************************************************************
    * Begin with making TowerClusters in 5x7 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters5x7(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 5x7 TowerClusters seeds finding
    caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu5x7; clu5x7.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.83) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu5x7.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu5x7.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu5x7.seedIeta = l1CaloTower.towerIeta;
                clu5x7.seedIphi = l1CaloTower.towerIphi;
                clu5x7.seedEta  = l1CaloTower.towerEta;
                clu5x7.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu5x7.seedIeta)<=13) { clu5x7.isBarrel = true;  }
                else if (abs(clu5x7.seedIeta)>=22) { clu5x7.isEndcap = true;  }
                else                               { clu5x7.isOverlap = true; }

                // Fill the TowerCluster towers
                clu5x7.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu5x7.totalEm      += l1CaloTower.towerEm;
                clu5x7.totalHad     += l1CaloTower.towerHad;
                clu5x7.totalEt      += l1CaloTower.towerEt;
                clu5x7.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu5x7.totalIem     += l1CaloTower.towerIem;
                clu5x7.totalIhad    += l1CaloTower.towerIhad;
                clu5x7.totalIet     += l1CaloTower.towerIet;
                clu5x7.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu5x7.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu5x7.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu5x7.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu5x7.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu5x7.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu5x7.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 4 && abs(d_iPhi) <= 6) || (abs(d_Eta) < 0.35 && abs(d_Phi) < 0.55)) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu5x7.nHits > 0.0) { l1TowerClusters5x7->push_back(clu5x7); }

    }  // end while loop of TowerClusters seeding

    // loop for 5x7 TowerClusters creation starting from the seed just found
    for (auto& clu5x7 : *l1TowerClusters5x7)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu5x7.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu5x7.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu5x7.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu5x7.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu5x7.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 5x7 towers mask
            if ((abs(d_iEta) <= 2 && abs(d_iPhi) <= 3) || (abs(d_Eta) < 0.2 && abs(d_Phi) < 0.3))
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu5x7.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu5x7.totalEm      += l1CaloTower.towerEm;
                clu5x7.totalHad     += l1CaloTower.towerHad;
                clu5x7.totalEt      += l1CaloTower.towerEt;
                clu5x7.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu5x7.totalIem     += l1CaloTower.towerIem;
                clu5x7.totalIhad    += l1CaloTower.towerIhad;
                clu5x7.totalIet     += l1CaloTower.towerIet;
                clu5x7.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu5x7.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu5x7.towerHits);
        clu5x7.InitHits(); clu5x7.towerHits = sortedHits;

        if (sortedHits.size() != 35) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 35 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

    }// end while loop of 5x7 TowerClusters creation


    /********************************************************************************************
    * Begin with making TowerClusters in 3x7 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters3x7(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 3x7 TowerClusters seeds finding
    caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu3x7; clu3x7.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.91) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu3x7.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu3x7.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu3x7.seedIeta = l1CaloTower.towerIeta;
                clu3x7.seedIphi = l1CaloTower.towerIphi;
                clu3x7.seedEta  = l1CaloTower.towerEta;
                clu3x7.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu3x7.seedIeta)<=13) { clu3x7.isBarrel = true;  }
                else if (abs(clu3x7.seedIeta)>=22) { clu3x7.isEndcap = true;  }
                else                               { clu3x7.isOverlap = true; }

                // Fill the TowerCluster towers
                clu3x7.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu3x7.totalEm      += l1CaloTower.towerEm;
                clu3x7.totalHad     += l1CaloTower.towerHad;
                clu3x7.totalEt      += l1CaloTower.towerEt;
                clu3x7.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu3x7.totalIem     += l1CaloTower.towerIem;
                clu3x7.totalIhad    += l1CaloTower.towerIhad;
                clu3x7.totalIet     += l1CaloTower.towerIet;
                clu3x7.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu3x7.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu3x7.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu3x7.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu3x7.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu3x7.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu3x7.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 2 && abs(d_iPhi) <= 6) || (abs(d_Eta) < 0.2 && abs(d_Phi) < 0.55)) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu3x7.nHits > 0.0) { l1TowerClusters3x7->push_back(clu3x7); }

    }  // end while loop of TowerClusters seeding

    // loop for 3x7 TowerClusters creation starting from the seed just found
    for (auto& clu3x7 : *l1TowerClusters3x7)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu3x7.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu3x7.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu3x7.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu3x7.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu3x7.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 3x7 towers mask
            if ((abs(d_iEta) <= 1 && abs(d_iPhi) <= 3) || (abs(d_Eta) < 0.13 && abs(d_Phi) < 0.3))
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu3x7.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu3x7.totalEm      += l1CaloTower.towerEm;
                clu3x7.totalHad     += l1CaloTower.towerHad;
                clu3x7.totalEt      += l1CaloTower.towerEt;
                clu3x7.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu3x7.totalIem     += l1CaloTower.towerIem;
                clu3x7.totalIhad    += l1CaloTower.towerIhad;
                clu3x7.totalIet     += l1CaloTower.towerIet;
                clu3x7.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu3x7.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu3x7.towerHits);
        clu3x7.InitHits(); clu3x7.towerHits = sortedHits;

        if (sortedHits.size() != 21) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 21 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

    }// end while loop of 3x7 TowerClusters creation


    /********************************************************************************************
    * Begin with making TowerClusters in 3x5 grid based on all energy not included in L1EG Objs.
    * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
    * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
    *********************************************************************************************/

    std::unique_ptr<TowerHelper::TowerClustersCollection> l1TowerClusters3x5(new TowerHelper::TowerClustersCollection);

    // re-initialize the stale flags
    for (auto &l1CaloTower : l1CaloTowers) { l1CaloTower.InitStale(); }

    // loop for 3x5 TowerClusters seeds finding
    caloTauSeedingFinished = false;
    while (!caloTauSeedingFinished)
    {
        TowerHelper::TowerCluster clu3x5; clu3x5.InitHits();

        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            if (abs(l1CaloTower.towerEta) > 2.91) { continue; }

            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale4seed) { continue; }

            // find highest ET tower and use to seed the TowerCluster
            if (clu3x5.nHits == 0.0)
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
                if (l1CaloTower.isBarrel) { clu3x5.barrelSeeded = true; }
                
                // Fill the seed tower variables
                clu3x5.seedIeta = l1CaloTower.towerIeta;
                clu3x5.seedIphi = l1CaloTower.towerIphi;
                clu3x5.seedEta  = l1CaloTower.towerEta;
                clu3x5.seedPhi  = l1CaloTower.towerPhi;
                if      (abs(clu3x5.seedIeta)<=13) { clu3x5.isBarrel = true;  }
                else if (abs(clu3x5.seedIeta)>=22) { clu3x5.isEndcap = true;  }
                else                               { clu3x5.isOverlap = true; }

                // Fill the TowerCluster towers
                clu3x5.towerHits.push_back(l1CaloTower);
                
                // Fill the TowerCluster overall variables
                clu3x5.totalEm      += l1CaloTower.towerEm;
                clu3x5.totalHad     += l1CaloTower.towerHad;
                clu3x5.totalEt      += l1CaloTower.towerEt;
                clu3x5.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu3x5.totalIem     += l1CaloTower.towerIem;
                clu3x5.totalIhad    += l1CaloTower.towerIhad;
                clu3x5.totalIet     += l1CaloTower.towerIet;
                clu3x5.totalEgIet   += l1CaloTower.l1egTowerIet;
                clu3x5.nHits++;

                continue;
            }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu3x5.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu3x5.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu3x5.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu3x5.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu3x5.seedPhi, l1CaloTower.towerPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= 2 && abs(d_iPhi) <= 4) || (abs(d_Eta) < 0.2 && abs(d_Phi) < 0.35)) { l1CaloTower.stale4seed = true; }
    
        } // end for loop over TPs

        if (clu3x5.nHits > 0.0) { l1TowerClusters3x5->push_back(clu3x5); }

    }  // end while loop of TowerClusters seeding

    // loop for 3x5 TowerClusters creation starting from the seed just found
    for (auto& clu3x5 : *l1TowerClusters3x5)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used by this clusters' mask
            if (l1CaloTower.stale) { continue; }

            // go on with unused l1CaloTowers which are not the initial seed
            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel
            if (clu3x5.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_diEta(clu3x5.seedIeta, l1CaloTower.towerIeta);
                d_iPhi = tower_diPhi(clu3x5.seedIphi, l1CaloTower.towerIphi);
            }
            // use eta/phi in HGCal
            else
            {
                d_Eta = clu3x5.seedEta - l1CaloTower.towerEta;
                d_Phi = reco::deltaPhi(clu3x5.seedPhi, l1CaloTower.towerPhi);
            }

            // cluster all towers in a 3x5 towers mask
            if ((abs(d_iEta) <= 1 && abs(d_iPhi) <= 2) || (abs(d_Eta) < 0.13 && abs(d_Phi) < 0.22))
            {
                l1CaloTower.stale = true;

                // Fill the TowerCluster towers
                clu3x5.towerHits.push_back(l1CaloTower);

                // Fill the TowerCluster overall variables
                clu3x5.totalEm      += l1CaloTower.towerEm;
                clu3x5.totalHad     += l1CaloTower.towerHad;
                clu3x5.totalEt      += l1CaloTower.towerEt;
                clu3x5.totalEgEt    += l1CaloTower.l1egTowerEt;
                clu3x5.totalIem     += l1CaloTower.towerIem;
                clu3x5.totalIhad    += l1CaloTower.towerIhad;
                clu3x5.totalIet     += l1CaloTower.towerIet;
                clu3x5.totalEgIet   += l1CaloTower.l1egTowerIet;
                if (l1CaloTower.towerIet > 0) clu3x5.nHits++;
            }
        }// end for loop of TP clustering

        // sort the TowerHits in the TowerCluster to have them organized as "a picture of it"
        std::vector<TowerHelper::TowerHit> sortedHits = sortPicLikeF(clu3x5.towerHits);
        clu3x5.InitHits(); clu3x5.towerHits = sortedHits;

        if (sortedHits.size() != 15) { std::cout << " ** WARNING : CLUSTER WITH WRONG NUMBER OF 15 TOWERS! (" << sortedHits.size() << " TOWERS FOUND)" << std::endl; }

    }// end while loop of 3x5 TowerClusters creation

    if (DEBUG) 
    {
        std::cout << "\n***************************************************************************************************************************************" << std::endl;
        std::cout << "***************************************************************************************************************************************\n" << std::endl;

        for (auto& clu9x9 : *l1TowerClusters9x9)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 9x9 seed " << " , eta " << clu9x9.seedIeta << " phi " << clu9x9.seedIphi << std::endl;
            std::cout << " -- clu 9x9 seed " << " , isBarrel " << clu9x9.isBarrel << " isEndcap " << clu9x9.isEndcap << " isOverlap " << clu9x9.isOverlap << std::endl;
            std::cout << " -- clu 9x9 towers etas (" << clu9x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerHits.size(); ++j) { std::cout  << ", " << clu9x9.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers phis (" << clu9x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerHits.size(); ++j) { std::cout << ", " << clu9x9.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers ems (" << clu9x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerHits.size(); ++j) { std::cout << ", " << clu9x9.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers hads (" << clu9x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerHits.size(); ++j) { std::cout << ", " << clu9x9.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 9x9 towers ets (" << clu9x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu9x9.towerHits.size(); ++j) { std::cout << ", " << clu9x9.towerHits[j].towerIet; }
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
            std::cout << " -- clu 7x7 towers etas (" << clu7x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerHits.size(); ++j) { std::cout  << ", " << clu7x7.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers phis (" << clu7x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerHits.size(); ++j) { std::cout << ", " << clu7x7.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers ems (" << clu7x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerHits.size(); ++j) { std::cout << ", " << clu7x7.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers hads (" << clu7x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerHits.size(); ++j) { std::cout << ", " << clu7x7.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 7x7 towers ets (" << clu7x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu7x7.towerHits.size(); ++j) { std::cout << ", " << clu7x7.towerHits[j].towerIet; }
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
            std::cout << " -- clu 5x5 towers etas (" << clu5x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerHits.size(); ++j) { std::cout  << ", " << clu5x5.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers phis (" << clu5x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerHits.size(); ++j) { std::cout << ", " << clu5x5.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers ems (" << clu5x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerHits.size(); ++j) { std::cout << ", " << clu5x5.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers hads (" << clu5x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerHits.size(); ++j) { std::cout << ", " << clu5x5.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x5 towers ets (" << clu5x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x5.towerHits.size(); ++j) { std::cout << ", " << clu5x5.towerHits[j].towerIet; }
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
            std::cout << " -- clu 5x9 towers etas (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout  << ", " << clu5x9.towerHits[j].towerEta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers phis (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout << ", " << clu5x9.towerHits[j].towerPhi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers ietas (" << clu5x9.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x9.towerHits.size(); ++j) { std::cout  << ", " << clu5x9.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x9 towers iphis (" << clu5x9.towerHits.size() << ") [";
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

        for (auto& clu5x7 : *l1TowerClusters5x7)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 5x7 seed " << " , eta " << clu5x7.seedIeta << " phi " << clu5x7.seedIphi << std::endl;
            std::cout << " -- clu 5x7 seed " << " , isBarrel " << clu5x7.isBarrel << " isEndcap " << clu5x7.isEndcap << " isOverlap " << clu5x7.isOverlap << std::endl;
            std::cout << " -- clu 5x7 towers etas (" << clu5x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x7.towerHits.size(); ++j) { std::cout  << ", " << clu5x7.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x7 towers phis (" << clu5x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x7.towerHits.size(); ++j) { std::cout << ", " << clu5x7.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x7 towers ems (" << clu5x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x7.towerHits.size(); ++j) { std::cout << ", " << clu5x7.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x7 towers hads (" << clu5x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x7.towerHits.size(); ++j) { std::cout << ", " << clu5x7.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x7 towers ets (" << clu5x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu5x7.towerHits.size(); ++j) { std::cout << ", " << clu5x7.towerHits[j].towerIet; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 5x7 number of towers " << clu5x7.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;

        for (auto& clu3x7 : *l1TowerClusters3x7)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 3x7 seed " << " , eta " << clu3x7.seedIeta << " phi " << clu3x7.seedIphi << std::endl;
            std::cout << " -- clu 3x7 seed " << " , isBarrel " << clu3x7.isBarrel << " isEndcap " << clu3x7.isEndcap << " isOverlap " << clu3x7.isOverlap << std::endl;
            std::cout << " -- clu 3x7 towers etas (" << clu3x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x7.towerHits.size(); ++j) { std::cout  << ", " << clu3x7.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x7 towers phis (" << clu3x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x7.towerHits.size(); ++j) { std::cout << ", " << clu3x7.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x7 towers ems (" << clu3x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x7.towerHits.size(); ++j) { std::cout << ", " << clu3x7.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x7 towers hads (" << clu3x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x7.towerHits.size(); ++j) { std::cout << ", " << clu3x7.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x7 towers ets (" << clu3x7.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x7.towerHits.size(); ++j) { std::cout << ", " << clu3x7.towerHits[j].towerIet; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x7 number of towers " << clu3x7.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;

        for (auto& clu3x5 : *l1TowerClusters3x5)
        {
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << " -- clu 3x5 seed " << " , eta " << clu3x5.seedIeta << " phi " << clu3x5.seedIphi << std::endl;
            std::cout << " -- clu 3x5 seed " << " , isBarrel " << clu3x5.isBarrel << " isEndcap " << clu3x5.isEndcap << " isOverlap " << clu3x5.isOverlap << std::endl;
            std::cout << " -- clu 3x5 towers etas (" << clu3x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x5.towerHits.size(); ++j) { std::cout  << ", " << clu3x5.towerHits[j].towerIeta; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x5 towers phis (" << clu3x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x5.towerHits.size(); ++j) { std::cout << ", " << clu3x5.towerHits[j].towerIphi; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x5 towers ems (" << clu3x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x5.towerHits.size(); ++j) { std::cout << ", " << clu3x5.towerHits[j].towerIem; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x5 towers hads (" << clu3x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x5.towerHits.size(); ++j) { std::cout << ", " << clu3x5.towerHits[j].towerIhad; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x5 towers ets (" << clu3x5.towerHits.size() << ") [";
            for (long unsigned int j = 0; j < clu3x5.towerHits.size(); ++j) { std::cout << ", " << clu3x5.towerHits[j].towerIet; }
            std::cout << "]" << std::endl;
            std::cout << " -- clu 3x5 number of towers " << clu3x5.nHits << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "*****************************************************************************************************************************************" << std::endl;
    }

    iEvent.put(std::move(l1TowerClusters9x9), "l1TowerClusters9x9");
    iEvent.put(std::move(l1TowerClusters7x7), "l1TowerClusters7x7");
    iEvent.put(std::move(l1TowerClusters5x5), "l1TowerClusters5x5");
    iEvent.put(std::move(l1TowerClusters5x9), "l1TowerClusters5x9");
    iEvent.put(std::move(l1TowerClusters5x7), "l1TowerClusters5x7");
    iEvent.put(std::move(l1TowerClusters3x7), "l1TowerClusters3x7");
    iEvent.put(std::move(l1TowerClusters3x5), "l1TowerClusters3x5");
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
    float eta_step = 0.08450;
    return floor(abs(eta)/eta_step) * std::copysign(1,eta);
}

std::vector<TowerHelper::TowerHit> CaloTowerHandler::sortPicLikeF(std::vector<TowerHelper::TowerHit> towerHits) const
{
    std::sort(begin(towerHits), end(towerHits), [](const TowerHelper::TowerHit &a, TowerHelper::TowerHit &b)
    { 
        // compute the difference in eta instead of direct comparison to account for possible little differences in the eta position
        // (mainly due to all the playing around that happens at the beginning to fill zeros and missing towers)
        if (abs(a.towerEta - b.towerEta) < 0.001)
        {
            if ((a.towerPhi < -2.4 && b.towerPhi > 2.4) || (b.towerPhi < -2.4 && a.towerPhi > 2.4))
            {
                if (a.towerPhi * b.towerPhi < 0) { return a.towerPhi > b.towerPhi; }
                else                             { return a.towerPhi < b.towerPhi; }
            }
            
            else { return a.towerPhi < b.towerPhi; }
        }
        else { return a.towerEta < b.towerEta; }
    });

    return towerHits;
}

std::vector<TowerHelper::TowerHit> CaloTowerHandler::sortPicLikeI(std::vector<TowerHelper::TowerHit> towerHits) const
{
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

DEFINE_FWK_MODULE(CaloTowerHandler);