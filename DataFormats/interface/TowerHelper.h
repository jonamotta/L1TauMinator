#ifndef TowerHelper_h
#define TowerHelper_h

#include <iostream>
#include <vector>

namespace TowerHelper 
{
    // class for single trigger tower hits (both barrel and endcap friendly)
    class TowerHit {
        public:
        float towerEta = 0.;
        float towerPhi = 0.;
        float towerEm = 0.;
        float towerHad = 0.;
        float towerEt = 0.;
        int   towerIeta = 0;
        int   towerIphi = 0;
        int   towerIem = 0;
        int   towerIhad = 0;
        int   towerIet = 0;
        bool  isBarrel = true;
        bool  stale = false;
        bool  stale4seed = false;

        // L1EG info
        float l1egTowerEt = 0.;
        int   l1egTowerIet = 0;
        int   nL1eg = 0;
        // int   l1egTrkSS = 0;
        // int   l1egTrkIso = 0;
        // int   l1egStandaloneSS = 0;
        // int   l1egStandaloneIso = 0;

        void InitStale()
        {
            stale = false;
            stale4seed = false;
        }
    };

    // class for NxN trigger towers clusters (both barrel and endcap friendly)
    class TowerCluster {
        public:
        bool  barrelSeeded = false;
        int   nHits = 0;
        int   seedIeta = -99;
        int   seedIphi = -99;
        float seedEta = -99;
        float seedPhi = -99;
        std::vector<float> towerEta;
        std::vector<float> towerPhi;
        std::vector<float> towerEm;
        std::vector<float> towerHad;
        std::vector<float> towerEt;
        std::vector<int>   towerIeta;
        std::vector<int>   towerIphi;
        std::vector<int>   towerIem;
        std::vector<int>   towerIhad;
        std::vector<int>   towerIet;

        float totalEm = 0;
        float totalHad = 0;
        float totalEt = 0;
        float totalIem = 0;
        float totalIhad = 0;
        float totalIet = 0;

        bool isBarrel = false;  // NxN TowerCluster fully contained in the barrel
        bool isOverlap = false; // NxN TowerCluster overlapping bewteen the barrel and the endcap
        bool isEndcap = false;  // NxN TowerCluster fully contained in the endcap

        int tauMatchIdx = -99;
        int jetMatchIdx = -99;

        void Init()
        {
            towerEta.clear();
            towerPhi.clear();
            towerEm.clear();
            towerHad.clear();
            towerEt.clear();
            towerIeta.clear();
            towerIphi.clear();
            towerIem.clear();
            towerIhad.clear();
            towerIet.clear();
        }
    };

    // collection of NxN trigger towers clusters (both barrel and endcap friendly)
    typedef std::vector<TowerCluster> TowerClustersCollection;
}
#endif