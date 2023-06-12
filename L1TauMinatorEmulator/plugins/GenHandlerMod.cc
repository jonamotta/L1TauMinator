#include <TLorentzVector.h>
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

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "L1TauMinator/DataFormats/interface/GenHelper.h"


typedef math::XYZTLorentzVector LorentzVector;

class GenHandlerMod : public edm::stream::EDProducer<> {
    public:
        explicit GenHandlerMod(const edm::ParameterSet&);

    private:
        //----edm control---
        void produce(edm::Event&, const edm::EventSetup&) override;

        //----private functions----
        bool isGoodTau(const reco::GenParticle& candidate) const;
        bool isStableLepton(const reco::GenParticle& daughter) const;
        bool isElectron(const reco::GenParticle& daughter) const;
        bool isMuon(const reco::GenParticle& daughter) const;
        bool isChargedHadron(const reco::GenParticle& daughter) const;
        bool isChargedHadronFromResonance(const reco::GenParticle& daughter) const;
        bool isNeutralPion(const reco::GenParticle& daughter) const;
        bool isNeutralPionFromResonance(const reco::GenParticle& daughter) const;
        bool isIntermediateResonance(const reco::GenParticle& daughter) const;
        bool isGamma(const reco::GenParticle& daughter) const;
        bool isStableNeutralHadron(const reco::GenParticle& daughter) const;
        bool isGoodBQuark(const reco::GenParticle& candidate) const;
        bool isFromId (const reco::Candidate& candidate, const int targetPdgId) const;
        bool CheckBit(const int number, const int bitpos) const;
        bool isFirstCopy (const reco::Candidate& candidate) const;

        //----tokens and handles----
        edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken;
        edm::Handle<reco::GenParticleCollection> genParticlesHandle;

        //----private variables----
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
GenHandlerMod::GenHandlerMod(const edm::ParameterSet& iConfig) 
    : genParticlesToken(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("GenParticles"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<GenHelper::GenTausCollection>("GenTausCollection");
}

void GenHandlerMod::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill the collection of good taus and their attributes
    std::unique_ptr<GenHelper::GenTausCollection> GenTausCollection(new GenHelper::GenTausCollection);

    iEvent.getByToken(genParticlesToken, genParticlesHandle);
    for (auto& particle : *genParticlesHandle.product())
    {
        GenHelper::GenTau GenTau;

        int AbsID = std::abs(particle.pdgId());
        int status = particle.status();

        if(AbsID == 12 || AbsID == 14 || AbsID == 16) { continue; }

        if(AbsID == 15 && status == 2) // DIFF : status==2 selection reduces taus by ~1/3
        {
            GenTau.pt = particle.pt();
            GenTau.eta = particle.eta();
            GenTau.phi = particle.phi();
            GenTau.e = particle.energy();
            GenTau.m = particle.mass();

            LorentzVector tau_p4vis(0., 0., 0., 0.);
            bool LeptonicDecay = false;
            int ID = particle.pdgId();

            for (int i = 0; i < (int)genParticlesHandle.product()->size(); i++)
            {
                reco::GenParticle candidate = genParticlesHandle.product()->at(i);

                // done like in here https://cmssdt.cern.ch/lxr/source/L1Trigger/L1TNtuples/plugins/L1GenTreeProducer.cc
                // POSSIBLE DIFF : to me it looks like I did it correctly but maybe I mis-interpreted how the variable is filled/read
                bool found = false;
                for (int j = 0; j < (int)candidate.numberOfMothers(); j++)
                {
                    int motherId = candidate.mother(j)->pdgId();
                    if (motherId == ID) { found = true; }
                }

                if (!found) { continue; }

                // DIFF : asking for direct decay product enlarges the number of passing candidates by ~5%
                // is last copy does not seem to make any difference
                // status==1 does not seem to make any difference
                if((candidate.pdgId() == 11 || candidate.pdgId() == -11) && candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() && candidate.status() == 1) { LeptonicDecay = true; }
                if((candidate.pdgId() == 13 || candidate.pdgId() == -13) && candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy() && candidate.status() == 1) { LeptonicDecay = true; }

                if(candidate.pdgId() == 12 || candidate.pdgId() == -12) { continue; }
                if(candidate.pdgId() == 14 || candidate.pdgId() == -14) { continue; }
                if(candidate.pdgId() == 16 || candidate.pdgId() == -16) { continue; }

                tau_p4vis = tau_p4vis + candidate.p4();
            }

            if (LeptonicDecay) { continue; }
            
            GenTau.visPt = tau_p4vis.Pt();
            GenTau.visEta = tau_p4vis.Eta();
            GenTau.visPhi = tau_p4vis.Phi();
            GenTau.visE = tau_p4vis.E();
            GenTau.visM = tau_p4vis.M();
            GenTau.visPtEm = -99.;
            GenTau.visEEm = -99.;
            GenTau.visPtHad = -99.;
            GenTau.visEHad = -99.;

            // if (abs(GenTau.visEta < 3.0)) { GenTausCollection->push_back(GenTau); }

            GenTausCollection->push_back(GenTau);
        }
    }

    iEvent.put(std::move(GenTausCollection),  "GenTausCollection");
}


DEFINE_FWK_MODULE(GenHandlerMod);