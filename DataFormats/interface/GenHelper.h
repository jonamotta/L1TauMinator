#ifndef GenHelper_h
#define GenHelper_h

#include <iostream>
#include <vector>

namespace GenHelper 
{
    class GenTau {
        public:
        float eta = 0.;
        float phi = 0.;
        float pt = 0.;
        float e = 0.;
        float m = 0.;
        float visEta = 0.;
        float visPhi = 0.;
        float visPt = 0.;
        float visE = 0.;
        float visM = 0.;
        float visPtEm = 0.;
        float visPtHad = 0.;
        float visEEm = 0.;
        float visEHad = 0.;
        int   DM = 0;
    };

    class GenJet {
        public:
        float eta = 0.;
        float phi = 0.;
        float pt = 0.;
        float e = 0.;
        float eEm = 0.;
        float eHad = 0.;
        float eInv = 0.;
    };

    typedef std::vector<GenTau> GenTausCollection;
    typedef std::vector<GenJet> GenJetsCollection;
}
#endif