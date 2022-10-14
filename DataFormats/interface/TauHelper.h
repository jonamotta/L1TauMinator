#ifndef TauHelper_h
#define TauHelper_h

#include <iostream>
#include <vector>

namespace TauHelper 
{
    class Tau {
        public:
        float eta = 0.;
        float phi = 0.;
        float pt = 0.;

        bool isBarrel = false;
        bool isEndcap = false;

        float IDscore = 0.;
    };

    typedef std::vector<Tau> TausCollection;
}
#endif