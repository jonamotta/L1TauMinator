#ifndef GenHelper_h
#define GenHelper_h

#include <iostream>
#include <vector>

namespace GenHelper 
{
    class GenTau {
        public:
        float eta = -99.;
        float phi = -99.;
        float pt = -99.;
        float e = -99.;
        float m = -99.;
        float visEta = -99.;
        float visPhi = -99.;
        float visPt = -99.;
        float visE = -99.;
        float visM = -99.;
        float visPtEm = -99.;
        float visPtHad = -99.;
        float visEEm = -99.;
        float visEHad = -99.;
        int   DM = -99;
    };

    class GenJet {
        public:
        float eta = -99.;
        float phi = -99.;
        float pt = -99.;
        float e = -99.;
        float eEm = -99.;
        float eHad = -99.;
        float eInv = -99.;
    };

    typedef std::vector<GenTau> GenTausCollection;
    typedef std::vector<GenJet> GenJetsCollection;
}
#endif