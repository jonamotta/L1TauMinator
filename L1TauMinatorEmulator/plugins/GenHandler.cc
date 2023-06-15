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

class GenHandler : public edm::stream::EDProducer<> {
    public:
        explicit GenHandler(const edm::ParameterSet&);

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
        
        edm::EDGetTokenT<reco::GenJetCollection> genJetsToken;
        edm::Handle<reco::GenJetCollection> genJetsHandle;

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
GenHandler::GenHandler(const edm::ParameterSet& iConfig) 
    : genParticlesToken(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("GenParticles"))),
      genJetsToken(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("GenJets"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<GenHelper::GenTausCollection>("GenTausCollection");
    produces<GenHelper::GenJetsCollection>("GenJetsCollection");
    produces<GenHelper::GenJetsCollection>("GenBJetsCollection");
}

void GenHandler::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill the collection of good taus and their attributes
    std::unique_ptr<GenHelper::GenTausCollection> GenTausCollection(new GenHelper::GenTausCollection);


    std::vector<reco::GenParticle> GenBQuarks;

    iEvent.getByToken(genParticlesToken, genParticlesHandle);
    for (auto& particle : *genParticlesHandle.product())
    {
        if (isGoodTau(particle))
        {

            GenHelper::GenTau GenTau;

            GenTau.pt = particle.pt();
            GenTau.eta = particle.eta();
            GenTau.phi = particle.phi();
            GenTau.e = particle.energy();
            GenTau.m = particle.mass();

            LorentzVector tau_p4vis(0., 0., 0., 0.);
            LorentzVector tau_p4em(0., 0., 0., 0.);
            LorentzVector tau_p4had(0., 0., 0., 0.);
            LorentzVector tau_p4mu(0., 0., 0., 0.); // used for debugging puproses only
            int n_pi = 0;
            int n_piZero = 0;
            int n_gamma = 0;
            int n_ele = 0;
            int n_mu = 0;

            // Loop over tau daughters to set DM and visible quantities
            const reco::GenParticleRefVector& daughters = particle.daughterRefVector();
            for (const auto& daughter : daughters) {

                if (isStableLepton(*daughter))
                {
                    if (isElectron(*daughter))
                    { 
                        n_ele++; 
                        tau_p4em += (daughter->p4());
                    }
                    else if (isMuon(*daughter))
                    { 
                        n_mu++;
                        tau_p4mu += (daughter->p4()); // used for debugging puproses only
                    }
                    tau_p4vis += (daughter->p4());
                }

                else if (isChargedHadron(*daughter))
                {
                    n_pi++;
                    tau_p4vis += (daughter->p4());
                    tau_p4had += (daughter->p4());
                }

                else if (isNeutralPion(*daughter))
                {
                    n_piZero++;
                    const reco::GenParticleRefVector& granddaughters = daughter->daughterRefVector();
                    for (const auto& granddaughter : granddaughters)
                    {
                        if (isGamma(*granddaughter))
                        {
                            n_gamma++;
                            tau_p4vis += (granddaughter->p4());
                            tau_p4em += (granddaughter->p4());
                        }
                    }
                }

                else if (isStableNeutralHadron(*daughter))
                {
                    tau_p4vis += (daughter->p4());
                    tau_p4had += (daughter->p4());
                }

                else
                {
                    const reco::GenParticleRefVector& granddaughters = daughter->daughterRefVector();
                    for (const auto& granddaughter : granddaughters)
                    {
                        if (isStableNeutralHadron(*granddaughter)) {
                            tau_p4vis += (granddaughter->p4());
                            tau_p4had += (granddaughter->p4());
                        }
                    }
                }

                /* Here the selection of the decay product according to the Pythia6 decayTree */
                if (isIntermediateResonance(*daughter))
                {
                    const reco::GenParticleRefVector& grandaughters = daughter->daughterRefVector();
                    for (const auto& grandaughter : grandaughters)
                    {
                        if (isChargedHadron(*grandaughter) || isChargedHadronFromResonance(*grandaughter))
                        {
                            n_pi++;
                            tau_p4vis += (grandaughter->p4());
                            tau_p4had += (grandaughter->p4());
                        }
                        else if (isNeutralPion(*grandaughter) || isNeutralPionFromResonance(*grandaughter))
                        {
                            n_piZero++;
                            const reco::GenParticleRefVector& descendants = grandaughter->daughterRefVector();
                            for (const auto& descendant : descendants)
                            {
                                if (isGamma(*descendant)) 
                                {
                                    n_gamma++;
                                    tau_p4vis += (descendant->p4());
                                    tau_p4em += (descendant->p4());
                                }
                            }
                        }
                    }
                }
            }

            GenTau.visPt = tau_p4vis.Pt();
            GenTau.visEta = tau_p4vis.Eta();
            GenTau.visPhi = tau_p4vis.Phi();
            GenTau.visE = tau_p4vis.E();
            GenTau.visM = tau_p4vis.M();
            GenTau.visPtEm = tau_p4em.Pt();
            GenTau.visEEm = tau_p4em.E();
            GenTau.visPtHad = tau_p4had.Pt();
            GenTau.visEHad = tau_p4had.E();

            // Leptonic tau decays
            if (n_pi == 0 && n_piZero == 0 && n_ele == 1)     { GenTau.DM = -1; }
            else if (n_pi == 0 && n_piZero == 0 && n_mu == 1) { GenTau.DM = -1; }
            // 1-prong tau decay
            else if (n_pi == 1 && n_piZero == 0) { GenTau.DM = 0; }
            // 1-prong + pi0 tau decay
            else if (n_pi == 1 && n_piZero == 1) { GenTau.DM = 1; }
            // 1-prong + pi0s tau decay
            else if (n_pi == 1 && n_piZero > 1)  { GenTau.DM = 2; }
            // 3-prongs tau decay
            else if (n_pi == 3 && n_piZero == 0) { GenTau.DM = 10; }
            // 3-prongs + pi0 tau decay
            else if (n_pi == 3 && n_piZero == 1) { GenTau.DM = 11; }
            // 3-prongs + pi0s tau decay
            else if (n_pi == 3 && n_piZero > 1)  { GenTau.DM = 12; }
            // other tau decays
            else { GenTau.DM = 100; }

            if (DEBUG)
            {
                printf(" - GEN TAU pt %f eta %f phi %f vispt %f viseta %f visphi %f DM %i\n",
                    GenTau.pt,
                    GenTau.eta,
                    GenTau.phi,
                    GenTau.visPt,
                    GenTau.visEta,
                    GenTau.visPhi,
                    GenTau.DM);
            }

            // skip taus out of HGcal acceptance and  non hadronic taus
            if (abs(GenTau.visEta < 3.0) && GenTau.DM > 0) { GenTausCollection->push_back(GenTau); }

        } // end if(goodTau())


        if (isGoodBQuark(particle))
        {
            // skip bs out of HGcal acceptance
            if (abs(particle.eta() < 3.0)) { GenBQuarks.push_back(particle); }
        }

    }

    // Create and Fill the collection of good jets and their attributes
    std::unique_ptr<GenHelper::GenJetsCollection> GenJetsCollection(new GenHelper::GenJetsCollection);
    std::unique_ptr<GenHelper::GenJetsCollection> GenBJetsCollection(new GenHelper::GenJetsCollection);

    iEvent.getByToken(genJetsToken, genJetsHandle);
    for (auto& jet : *genJetsHandle.product())
    {
        // skip very forward jets
        if (abs(jet.eta()) > 3.5) { continue; }
        // skip very low pt jets
        if (jet.pt() < 15) { continue; }

        GenHelper::GenJet GenJet;

        GenJet.pt   = jet.pt();
        GenJet.eta  = jet.eta();
        GenJet.phi  = jet.phi();
        GenJet.e    = jet.energy();
        GenJet.eEm  = jet.emEnergy();
        GenJet.eHad = jet.hadEnergy();
        GenJet.eInv = jet.invisibleEnergy();

        if (DEBUG)
        {
            printf(" - GEN JET pt %f eta %f phi %f e %f eEM %f eHad %f eInv %f\n",
                    GenJet.pt,
                    GenJet.eta,
                    GenJet.phi,
                    GenJet.e,
                    GenJet.eEm,
                    GenJet.eHad,
                    GenJet.eInv);
        }

        // Remove overlap between jets and taus (keep tau remove jet if dR(dR2)<=0.1(0.01) with visibe tau component)
        bool skipTauJet = false;
        for (auto const& tau : *GenTausCollection)
        {
            float dEta = GenJet.eta - tau.visEta;
            float dPhi = reco::deltaPhi(GenJet.phi, tau.visPhi);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 < 0.01)
            {
                skipTauJet = true;

                if (DEBUG)
                {
                    printf("        - GEN TAU vispt %f viseta %f visphi %f DM %i - dR2 %f\n",
                        tau.visPt,
                        tau.visEta,
                        tau.visPhi,
                        tau.DM,
                        dR2);
                }
            }
        }

        bool bJet = false;
        float maxPt = -99.;
        for (long unsigned int idx = 0; idx < GenBQuarks.size(); idx++)
        {
            const reco::GenParticle& b = GenBQuarks[idx];

            float dEta = GenJet.eta - b.eta();
            float dPhi = reco::deltaPhi(GenJet.phi, b.phi());
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (dR2 < 0.25 && b.pt() > maxPt)
            {
                bJet = true;
                maxPt = b.pt();
            } 
        }

        if (!skipTauJet)
        {
            if (bJet) { GenBJetsCollection->push_back(GenJet); }
            else      { GenJetsCollection->push_back(GenJet); }
        }
    }

    iEvent.put(std::move(GenTausCollection),  "GenTausCollection");
    iEvent.put(std::move(GenJetsCollection),  "GenJetsCollection");
    iEvent.put(std::move(GenBJetsCollection), "GenBJetsCollection");
}

bool GenHandler::isGoodTau(const reco::GenParticle& candidate) const
{
  return (std::abs(candidate.pdgId()) == 15 && candidate.status() == 2);
}

bool GenHandler::isChargedHadron(const reco::GenParticle& candidate) const
{
  return ((std::abs(candidate.pdgId()) == 211 || std::abs(candidate.pdgId()) == 321) && candidate.status() == 1 &&
          candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool GenHandler::isChargedHadronFromResonance(const reco::GenParticle& candidate) const
{
  return ((std::abs(candidate.pdgId()) == 211 || std::abs(candidate.pdgId()) == 321) && candidate.status() == 1 &&
          candidate.isLastCopy());
}

bool GenHandler::isStableLepton(const reco::GenParticle& candidate) const
{
  return ((std::abs(candidate.pdgId()) == 11 || std::abs(candidate.pdgId()) == 13) && candidate.status() == 1 &&
          candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool GenHandler::isElectron(const reco::GenParticle& candidate) const
{
  return (std::abs(candidate.pdgId()) == 11 && candidate.isDirectPromptTauDecayProductFinalState() &&
          candidate.isLastCopy());
}

bool GenHandler::isMuon(const reco::GenParticle& candidate) const
{
  return (std::abs(candidate.pdgId()) == 13 && candidate.isDirectPromptTauDecayProductFinalState() &&
          candidate.isLastCopy());
}

bool GenHandler::isNeutralPion(const reco::GenParticle& candidate) const
{
  return (std::abs(candidate.pdgId()) == 111 && candidate.status() == 2 &&
          candidate.statusFlags().isTauDecayProduct() && !candidate.isDirectPromptTauDecayProductFinalState());
}

bool GenHandler::isNeutralPionFromResonance(const reco::GenParticle& candidate) const
{
  return (std::abs(candidate.pdgId()) == 111 && candidate.status() == 2 && candidate.statusFlags().isTauDecayProduct());
}

bool GenHandler::isGamma(const reco::GenParticle& candidate) const
{
  return (std::abs(candidate.pdgId()) == 22 && candidate.status() == 1 && candidate.statusFlags().isTauDecayProduct() &&
          !candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool GenHandler::isIntermediateResonance(const reco::GenParticle& candidate) const
{
  return ((std::abs(candidate.pdgId()) == 213 || std::abs(candidate.pdgId()) == 20213 ||
           std::abs(candidate.pdgId()) == 24) &&
          candidate.status() == 2);
}

bool GenHandler::isStableNeutralHadron(const reco::GenParticle& candidate) const
{
  return (!(std::abs(candidate.pdgId()) > 10 && std::abs(candidate.pdgId()) < 17) && !isChargedHadron(candidate) &&
          candidate.status() == 1);
}

bool GenHandler::isGoodBQuark(const reco::GenParticle& candidate) const
{
    // int status_flags = candidate.userInt("generalGenFlags"); // only exists for pat::GenParticle
    return (std::abs(candidate.pdgId()) == 5 && /*candidate.status() == 21 && CheckBit(status_flags, 12) && CheckBit(status_flags, 8) &&*/ isFromId(candidate, 25));
    //                                             incoming particle            is first copy                from hard process                 from Higgs decay
}

bool GenHandler::isFromId (const reco::Candidate& candidate, const int targetPdgId) const
{
    if (abs(candidate.pdgId()) == targetPdgId)
    { 
        if(abs(candidate.pdgId()) == 5) { return isFirstCopy(candidate); }
        else                            { return true; }
    }

    for (unsigned int motherIdx = 0; motherIdx < candidate.numberOfMothers(); motherIdx++)
    {
        const reco::Candidate& mother = *candidate.mother(motherIdx);
        if (isFromId(mother, targetPdgId)) { return true; }
    }
    
    // no mother found
    return false;
}

bool GenHandler::CheckBit(const int number, const int bitpos) const
{
  bool res = number & (1 << bitpos);
  return res;
}

bool GenHandler::isFirstCopy (const reco::Candidate& candidate) const
{
    int cloneIdx = -1;
    int id = candidate.pdgId();
    for (unsigned int motherIdx = 0; motherIdx < candidate.numberOfMothers(); motherIdx++)
    {
        const reco::Candidate& mother = *candidate.mother(motherIdx);
        if (mother.pdgId() == id)
        {
            cloneIdx = motherIdx;
            break;
        }
    }
    
    if (cloneIdx == -1) { return true; }
    else                { return false; };
    
}


DEFINE_FWK_MODULE(GenHandler);