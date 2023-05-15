#ifndef TowerHelper_h
#define TowerHelper_h

#include <iostream>
#include <vector>

namespace TowerHelper 
{
    // class for single trigger tower hits (barrel, endcap, and hf friendly)
    class TowerHit {
        public:
        float towerEta = -99.;
        float towerPhi = -99.;
        float towerEm = -99.;
        float towerHad = -99.;
        float towerEt = -99.;
        int   towerIeta = -99;
        int   towerIphi = -99;
        int   towerIem = -99;
        int   towerIhad = -99;
        int   towerIet = -99;
        bool  isBarrel = true;
        bool  stale = false;
        bool  stale4seed = false;

        // L1EG info
        float l1egTowerEt = -99.;
        int   l1egTowerIet = -99;
        int   nL1eg = 0;
        // int   l1egTrkSS = -99;
        // int   l1egTrkIso = -99;
        // int   l1egStandaloneSS = -99;
        // int   l1egStandaloneIso = -99;

        void InitStale()
        {
            stale = false;
            stale4seed = false;
        }
    };

    // class for NxN trigger towers clusters (barrel, endcap, and hf friendly)
    class TowerCluster {
        public:
        bool  barrelSeeded = false;
        int   nHits = 0;
        int   seedIeta = -99;
        int   seedIphi = -99;
        float seedEta = -99.;
        float seedPhi = -99.;

        float totalEm = 0.0;
        float totalHad = 0.0;
        float totalEt = 0.0;
        float totalEgEt = 0.0;
        int   totalIem = 0;
        int   totalIhad = 0;
        int   totalIet = 0;
        int   totalEgIet = 0;

        bool isBarrel = false;  // NxM TowerCluster fully contained in the barrel
        bool isOverlap = false; // NxM TowerCluster overlapping bewteen the barrel and the endcap
        bool isEndcap = false;  // NxM TowerCluster fully contained in the endcap

        int tauMatchIdx = -99;
        int jetMatchIdx = -99;
        int cl3dMatchIdx = -99;

        bool isPhiFlipped = false;

        float IDscore = -99.;
        float calibPt = -99.;

        std::vector<TowerHit> towerHits;

        void InitHits() { towerHits.clear(); }
    };

    // collection of NxN trigger towers clusters (barrel, endcap, and hf friendly)
    typedef std::vector<TowerCluster> TowerClustersCollection;
}
#endif