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
// #include "L1TauMinator/L1TauMinatorEmulator/interface/GenHelper.h"

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
}

void GenHandler::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill the collection of good taus and their attributes
    std::unique_ptr<GenHelper::GenTausCollection> GenTausCollection(new GenHelper::GenTausCollection);

    iEvent.getByToken(genParticlesToken, genParticlesHandle);
    for (auto& particle : *genParticlesHandle.product())
    {
        if (!isGoodTau(particle)) { continue; }

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
        if (n_pi == 0 && n_piZero == 0 && n_ele == 1)     { GenTau.DM = 11; }
        else if (n_pi == 0 && n_piZero == 0 && n_mu == 1) { GenTau.DM = 13; }
        // 1-prong tau decay
        else if (n_pi == 1 && n_piZero == 0) { GenTau.DM = 0; }
        // 1-prong + pi0s tau decay
        else if (n_pi == 1 && n_piZero >= 1) { GenTau.DM = 1; }
        // 3-prongs tau decay
        else if (n_pi == 3 && n_piZero == 0) { GenTau.DM = 4; }
        // 3-prongs + pi0s tau decay
        else if (n_pi == 3 && n_piZero >= 1) { GenTau.DM = 5; }
        // other tau decays
        else { GenTau.DM = -1; }

        if (DEBUG)
        {
            std::cout << "---------------------------------------------------------------------" <<std::endl;
            std::cout << " gentau (DM "<< GenTau.DM <<"): pt " <<GenTau.visPt << " e " << GenTau.visE<< std::endl;
            std::cout << " gentau (DM "<< GenTau.DM <<"): ptEm " << GenTau.visPtEm << " ptHad " << GenTau.visPtHad << " ptMu " << tau_p4mu.Pt() << " eEm " << GenTau.visEEm << " eHad " << GenTau.visEHad << " eMu" << tau_p4mu.E() << std::endl;
            std::cout << "---------------------------------------------------------------------" <<std::endl;
        }

        GenTausCollection->push_back(GenTau);

    }

    // Create and Fill the collection of good jets and their attributes
    std::unique_ptr<GenHelper::GenJetsCollection> GenJetsCollection(new GenHelper::GenJetsCollection);

    iEvent.getByToken(genJetsToken, genJetsHandle);
    for (auto& jet : *genJetsHandle.product())
    {
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
            std::cout << "---------------------------------------------------------------------" <<std::endl;
            std::cout << "pt :              in " << jet.pt()              << " out " << GenJet.pt   << std::endl;
            std::cout << "eta :             in " << jet.eta()             << " out " << GenJet.eta  << std::endl;
            std::cout << "phi :             in " << jet.phi()             << " out " << GenJet.phi  << std::endl;
            std::cout << "energy :          in " << jet.energy()          << " out " << GenJet.e    << std::endl;
            std::cout << "emEnergy :        in " << jet.emEnergy()        << " out " << GenJet.eEm  << std::endl;
            std::cout << "hadEnergy :       in " << jet.hadEnergy()       << " out " << GenJet.eHad << std::endl;
            std::cout << "invisibleEnergy : in " << jet.invisibleEnergy() << " out " << GenJet.eInv << std::endl;
            std::cout << "---------------------------------------------------------------------" <<std::endl;
        }

        GenJetsCollection->push_back(GenJet);
    }
    
    // FIXME : not sure if this overlap removal is a good thing or a bad thing
    /*
    // Remove overlap between jets and taus (keep tau remove jet if dR<=0.4 with visibe tau component)
    for (long unsigned int i = 0; i < GenTausCollection.size(); i++)
    {
        TLorentzVector tau_p4(0.,0.,0.,0.);
        tau_p4.SetPtEtaPhiE(GenTausCollection[i].visPt, GenTausCollection[i].visEta, GenTausCollection[i].visPhi, GenTausCollection[i].visE);

        for (long unsigned int j = 0; j < GenJetsCollection.size(); j++)
        {
            TLorentzVector jet_p4(0.,0.,0.,0.);
            jet_p4.SetPtEtaPhiE(GenJetsCollection[i].pt, GenJetsCollection[i].eta, GenJetsCollection[i].phi, GenJetsCollection[i].e);

            if (jet_p4.DeltaR(tau_p4)<=0.4) { GenJetsCollection.erase(GenJetsCollection.begin()+j); }
        }
    }
    */

    iEvent.put(std::move(GenTausCollection), "GenTausCollection");
    iEvent.put(std::move(GenJetsCollection), "GenJetsCollection");
}

bool GenHandler::isGoodTau(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 15 && candidate.status() == 2);
}

bool GenHandler::isChargedHadron(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 211 || std::abs(candidate.pdgId()) == 321) && candidate.status() == 1 &&
          candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool GenHandler::isChargedHadronFromResonance(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 211 || std::abs(candidate.pdgId()) == 321) && candidate.status() == 1 &&
          candidate.isLastCopy());
}

bool GenHandler::isStableLepton(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 11 || std::abs(candidate.pdgId()) == 13) && candidate.status() == 1 &&
          candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool GenHandler::isElectron(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 11 && candidate.isDirectPromptTauDecayProductFinalState() &&
          candidate.isLastCopy());
}

bool GenHandler::isMuon(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 13 && candidate.isDirectPromptTauDecayProductFinalState() &&
          candidate.isLastCopy());
}

bool GenHandler::isNeutralPion(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 111 && candidate.status() == 2 &&
          candidate.statusFlags().isTauDecayProduct() && !candidate.isDirectPromptTauDecayProductFinalState());
}

bool GenHandler::isNeutralPionFromResonance(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 111 && candidate.status() == 2 && candidate.statusFlags().isTauDecayProduct());
}

bool GenHandler::isGamma(const reco::GenParticle& candidate) const {
  return (std::abs(candidate.pdgId()) == 22 && candidate.status() == 1 && candidate.statusFlags().isTauDecayProduct() &&
          !candidate.isDirectPromptTauDecayProductFinalState() && candidate.isLastCopy());
}

bool GenHandler::isIntermediateResonance(const reco::GenParticle& candidate) const {
  return ((std::abs(candidate.pdgId()) == 213 || std::abs(candidate.pdgId()) == 20213 ||
           std::abs(candidate.pdgId()) == 24) &&
          candidate.status() == 2);
}

bool GenHandler::isStableNeutralHadron(const reco::GenParticle& candidate) const {
  return (!(std::abs(candidate.pdgId()) > 10 && std::abs(candidate.pdgId()) < 17) && !isChargedHadron(candidate) &&
          candidate.status() == 1);
}

DEFINE_FWK_MODULE(GenHandler);