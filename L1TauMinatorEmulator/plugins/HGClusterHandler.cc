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

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"

#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
// #include "L1TauMinator/L1TauMinatorEmulator/interface/HGClusterHelper.h"

class HGClusterHandler : public edm::stream::EDProducer<> {
    public:
        explicit HGClusterHandler(const edm::ParameterSet&);

    private:
        //----edm control---
        void produce(edm::Event&, const edm::EventSetup&) override;

        //----private functions----

        //----tokens and handles----
        edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> HGClusterToken;
        edm::Handle<l1t::HGCalMulticlusterBxCollection> HGClusterHandle;

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
HGClusterHandler::HGClusterHandler(const edm::ParameterSet& iConfig) 
    : HGClusterToken(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalClusters"))),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<HGClusterHelper::HGClustersCollection>("HGClustersCollection");
}

void HGClusterHandler::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{   
    // Create and Fill the collection of 3D clusters and their usefull attributes
    std::unique_ptr<HGClusterHelper::HGClustersCollection> HGClustersCollection(new HGClusterHelper::HGClustersCollection);

    iEvent.getByToken(HGClusterToken, HGClusterHandle);
    for (auto& cl3d : *HGClusterHandle.product())
    {
        HGClusterHelper::HGCluster HGCluster;

        HGCluster.pt = cl3d.pt();
        HGCluster.energy = cl3d.energy();
        HGCluster.eta = cl3d.eta();
        HGCluster.phi = cl3d.phi();
        HGCluster.showerlength = cl3d.showerLength();
        HGCluster.coreshowerlength = cl3d.coreShowerLength();
        HGCluster.firstlayer = cl3d.firstLayer();
        HGCluster.seetot = cl3d.sigmaEtaEtaTot();
        HGCluster.seemax = cl3d.sigmaEtaEtaMax();
        HGCluster.spptot = cl3d.sigmaPhiPhiTot();
        HGCluster.sppmax = cl3d.sigmaPhiPhiMax();
        HGCluster.szz = cl3d.sigmaZZ();
        HGCluster.srrtot = cl3d.sigmaRRTot();
        HGCluster.srrmax = cl3d.sigmaRRMax();
        HGCluster.srrmean = cl3d.sigmaRRMean();
        HGCluster.hoe = cl3d.hOverE();
        HGCluster.meanz = cl3d.zBarycenter();
        HGCluster.quality = cl3d.hwQual();

        if (DEBUG)
        {
            std::cout << "---------------------------------------------------------------------" <<std::endl;
            std::cout << "pt :               in " << cl3d.pt()               << " out " << HGCluster.pt               << std::endl;
            std::cout << "energy :           in " << cl3d.energy()           << " out " << HGCluster.energy           << std::endl;
            std::cout << "eta :              in " << cl3d.eta()              << " out " << HGCluster.eta              << std::endl;
            std::cout << "phi :              in " << cl3d.phi()              << " out " << HGCluster.phi              << std::endl;
            std::cout << "showerlength :     in " << cl3d.showerLength()     << " out " << HGCluster.showerlength     << std::endl;
            std::cout << "coreshowerlength : in " << cl3d.coreShowerLength() << " out " << HGCluster.coreshowerlength << std::endl;
            std::cout << "firstlayer :       in " << cl3d.firstLayer()       << " out " << HGCluster.firstlayer       << std::endl;
            std::cout << "seetot :           in " << cl3d.sigmaEtaEtaTot()   << " out " << HGCluster.seetot           << std::endl;
            std::cout << "seemax :           in " << cl3d.sigmaEtaEtaMax()   << " out " << HGCluster.seemax           << std::endl;
            std::cout << "spptot :           in " << cl3d.sigmaPhiPhiTot()   << " out " << HGCluster.spptot           << std::endl;
            std::cout << "sppmax :           in " << cl3d.sigmaPhiPhiMax()   << " out " << HGCluster.sppmax           << std::endl;
            std::cout << "szz :              in " << cl3d.sigmaZZ()          << " out " << HGCluster.szz              << std::endl;
            std::cout << "srrtot :           in " << cl3d.sigmaRRTot()       << " out " << HGCluster.srrtot           << std::endl;
            std::cout << "srrmax :           in " << cl3d.sigmaRRMax()       << " out " << HGCluster.srrmax           << std::endl;
            std::cout << "srrmean :          in " << cl3d.sigmaRRMean()      << " out " << HGCluster.srrmean          << std::endl;
            std::cout << "hoe :              in " << cl3d.hOverE()           << " out " << HGCluster.hoe              << std::endl;
            std::cout << "meanz :            in " << cl3d.zBarycenter()      << " out " << HGCluster.meanz            << std::endl;
            std::cout << "quality :          in " << cl3d.hwQual()           << " out " << HGCluster.quality          << std::endl;
            std::cout << "---------------------------------------------------------------------" <<std::endl;
        }

        HGClustersCollection->push_back(HGCluster);
    }

    iEvent.put(std::move(HGClustersCollection), "HGClustersCollection");
}

DEFINE_FWK_MODULE(HGClusterHandler);