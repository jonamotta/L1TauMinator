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
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "L1TauMinator/DataFormats/interface/GenHelper.h"


typedef math::XYZTLorentzVector LorentzVector;

class GenHandlerMenu : public edm::stream::EDProducer<> {
    public:
        explicit GenHandlerMenu(const edm::ParameterSet&);

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
GenHandlerMenu::GenHandlerMenu(const edm::ParameterSet& iConfig) 
    : genParticlesToken(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("GenParticles"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<GenHelper::GenTausCollection>("GenTausCollection");
}

void GenHandlerMenu::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill the collection of good taus and their attributes
    std::unique_ptr<GenHelper::GenTausCollection> GenTausCollection(new GenHelper::GenTausCollection);

    // supporting vectors for the definition of hadronic taus
    // copied from: https://cmssdt.cern.ch/lxr/source/L1Trigger/L1TNtuples/plugins/L1GenTreeProducer.cc#0143
    std::vector<int> allPartStatus;
    std::vector<int> allPartId;
    std::vector<int> allPartParentId;
    std::vector<LorentzVector> allPartP4;
    iEvent.getByToken(genParticlesToken, genParticlesHandle);
    for (auto& p : *genParticlesHandle.product())
    {
        int id = p.pdgId();
 
        // See if the parent was interesting
        int parentID = -10000;
        unsigned int nMo = p.numberOfMothers();
        for (unsigned int i = 0; i < nMo; ++i) {
          int thisParentID = dynamic_cast<const reco::GenParticle*>(p.mother(i))->pdgId();
          //
          // Is this a bottom hadron?
          int hundredsIndex = abs(thisParentID) / 100;
          int thousandsIndex = abs(thisParentID) / 1000;
          if (((abs(thisParentID) >= 23) && (abs(thisParentID) <= 25)) || (abs(thisParentID) == 6) ||
              (hundredsIndex == 5) || (hundredsIndex == 4) || (thousandsIndex == 5) || (thousandsIndex == 4))
            parentID = thisParentID;
        }
        if ((parentID == -10000) && (nMo > 0))
          parentID = dynamic_cast<const reco::GenParticle*>(p.mother(0))->pdgId();
        //
        // If the parent of this particle is interesting, store all of the info
        if ((parentID != p.pdgId()) &&
            ((parentID > -9999) || (abs(id) == 11) || (abs(id) == 13) || (abs(id) == 23) || (abs(id) == 24) ||
             (abs(id) == 25) || (abs(id) == 4) || (abs(id) == 5) || (abs(id) == 6))) {

            allPartStatus.push_back(p.status());
            allPartId.push_back(p.pdgId());
            allPartParentId.push_back(parentID);
            allPartP4.push_back(p.p4());
        }
    }


    std::vector<GenHelper::GenTau> AllTaus;

    // loop to create hadronic tau candidates' visible 4-vectors
    iEvent.getByToken(genParticlesToken, genParticlesHandle);
    for (auto& particle : *genParticlesHandle.product())
    {
        GenHelper::GenTau GenTau;

        int AbsID = std::abs(particle.pdgId());
        int status = particle.status();

        if(AbsID == 12 || AbsID == 14 || AbsID == 16) { continue; }

        if(AbsID == 15 && status == 2)
        {
            GenTau.pt = particle.pt();
            GenTau.eta = particle.eta();
            GenTau.phi = particle.phi();
            GenTau.e = particle.energy();
            GenTau.m = particle.mass();

            LorentzVector tau_p4vis(0., 0., 0., 0.);
            LorentzVector tau_p4(0., 0., 0., 0.);
            bool LeptonicDecay = false;
            int ID = particle.pdgId();

            for (int i = 0; i < (int)allPartStatus.size(); i++)
            {
                int id = allPartId[i];
                int parentID = allPartParentId[i];
                LorentzVector P4 = allPartP4[i];
                
                if (parentID != ID) { continue; }

                if((id == 11 || id == -11)) { LeptonicDecay = true; }
                if((id == 13 || id == -13)) { LeptonicDecay = true; }

                tau_p4 = tau_p4 + P4;

                if(id == 12 || id == -12) { continue; }
                if(id == 14 || id == -14) { continue; }
                if(id == 16 || id == -16) { continue; }

                tau_p4vis = tau_p4vis + P4;

            }

            if (LeptonicDecay) { continue; }
            
            GenTau.visPt = tau_p4vis.Pt();
            GenTau.visEta = tau_p4vis.Eta();
            GenTau.visPhi = tau_p4vis.Phi();
            GenTau.sumPt = tau_p4.Pt();
            GenTau.sumEta = tau_p4.Eta();
            GenTau.sumPhi = tau_p4.Phi();
            GenTau.visE = tau_p4vis.E();
            GenTau.visM = tau_p4vis.M();
            GenTau.visPtEm = -99.;
            GenTau.visEEm = -99.;
            GenTau.visPtHad = -99.;
            GenTau.visEHad = -99.;

            if (abs(GenTau.visEta) < 3.0) { AllTaus.push_back(GenTau); }
        }
    }


    for (int j = 0; j < (int)AllTaus.size(); j++)
    {
        GenHelper::GenTau GenTau = AllTaus[j];

        double Isolation = 0.;

        for (int i = 0; i < (int)allPartStatus.size(); i++)
        {
            int id = allPartId[i];
            LorentzVector P4 = allPartP4[i];

            if (allPartStatus[i] != 1) { continue; } // use only final state particles
            if(id == 12 || id == -12) { continue; }
            if(id == 14 || id == -14) { continue; }
            if(id == 16 || id == -16) { continue; }

            double dEta = GenTau.visEta - P4.Eta();
            double dPhi = reco::deltaPhi(GenTau.visPhi, P4.Phi());
            double dR2 = dEta*dEta + dPhi*dPhi;
            
            if (dR2 < 0.09) { Isolation += P4.Pt(); }
        }

        Isolation  = Isolation / GenTau.visPt - 1.;

        if (Isolation < 0.15) { GenTausCollection->push_back(GenTau); }
    }

    iEvent.put(std::move(GenTausCollection),  "GenTausCollection");
}


DEFINE_FWK_MODULE(GenHandlerMenu);