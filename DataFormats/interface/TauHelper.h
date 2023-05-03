#ifndef TauHelper_h
#define TauHelper_h

#include <iostream>
#include <vector>

namespace TauHelper 
{
    class Tau {
        public:
        float eta = -99.;
        float phi = -99.;
        float pt = -99.;

        bool isBarrel = false;
        bool isEndcap = false;

        float IDscore = -99.;
    };

    typedef std::vector<Tau> TausCollection;
}
#endif