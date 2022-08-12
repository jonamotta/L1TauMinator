#ifndef HGClusterHelper_h
#define HGClusterHelper_h

#include <iostream>
#include <vector>

namespace HGClusterHelper 
{
    class HGCluster {
        public:
        float pt = 0.;
        float energy = 0.;
        float eta = 0.;
        float phi = 0.;
        float showerlength = 0.;
        float coreshowerlength = 0.;
        float firstlayer = 0.;
        float seetot = 0.;
        float seemax = 0.;
        float spptot = 0.;
        float sppmax = 0.;
        float szz = 0.;
        float srrtot = 0.;
        float srrmax = 0.;
        float srrmean = 0.;
        float hoe = 0.;
        float meanz = 0.;
        int   quality = 0;

        int tauMatchIdx = -99;
        int jetMatchIdx = -99;
    };

    typedef std::vector<HGCluster> HGClustersCollection;
}
#endif