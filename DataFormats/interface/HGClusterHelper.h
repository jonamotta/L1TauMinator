#ifndef HGClusterHelper_h
#define HGClusterHelper_h

#include <iostream>
#include <vector>

namespace HGClusterHelper 
{
    class HGCluster {
        public:
        float pt = -99.;
        float energy = -99.;
        float eta = -99.;
        float phi = -99.;
        float showerlength = -99.;
        float coreshowerlength = -99.;
        float firstlayer = -99.;
        float seetot = -99.;
        float seemax = -99.;
        float spptot = -99.;
        float sppmax = -99.;
        float szz = -99.;
        float srrtot = -99.;
        float srrmax = -99.;
        float srrmean = -99.;
        float hoe = -99.;
        float meanz = -99.;
        int   quality = -99;
        int   puId = -1;
        int   pionId = -1;
        float puIdScore = -99;
        float pionIdScore = -99;

        int tauMatchIdx = -99;
        int jetMatchIdx = -99;

        float IDscore = -99.;
        float calibPt = -99.;

        bool stale = false;
    };

    typedef std::vector<HGCluster> HGClustersCollection;
}
#endif