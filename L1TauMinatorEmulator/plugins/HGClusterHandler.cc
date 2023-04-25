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
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterEgID.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"


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
        enum class UseEmInterp { No, EmOnly, AllKeepHad, AllKeepTot };
        UseEmInterp scenario;
        StringCutObjectSelector<l1t::HGCalMulticluster> preEmId;
        l1tpf::HGC3DClusterEgID VsPionId;
        l1tpf::HGC3DClusterEgID VsPuId;
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
      scenario(UseEmInterp::No),
      preEmId(iConfig.getParameter<std::string>("preEmId")),
      VsPionId(iConfig.getParameter<edm::ParameterSet>("VsPionId")),
      VsPuId(iConfig.getParameter<edm::ParameterSet>("VsPuId")),
      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    
    produces<HGClusterHelper::HGClustersCollection>("HGClustersCollection");

    if (!VsPionId.method().empty()) {
      VsPionId.prepareTMVA();
    }
    if (!VsPuId.method().empty()) {
      VsPuId.prepareTMVA();
    }
}

void HGClusterHandler::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill the collection of 3D clusters and their usefull attributes
    std::unique_ptr<HGClusterHelper::HGClustersCollection> HGClustersCollection(new HGClusterHelper::HGClustersCollection);

    iEvent.getByToken(HGClusterToken, HGClusterHandle);
    for (auto cl3dIt = HGClusterHandle->begin(0); cl3dIt != HGClusterHandle->end(0); ++cl3dIt)
    {
        auto& cl3d = *cl3dIt;
        HGClusterHelper::HGCluster HGCluster;

        // IMPLEMENT PU/PION ID AS DONE IN https://github.com/cms-sw/cmssw/blob/master/L1Trigger/Phase2L1ParticleFlow/plugins/PFClusterProducerFromHGC3DClusters.cc#L120
        bool isEM = preEmId(*cl3dIt);
        l1t::PFCluster cluster(cl3d.pt(), cl3d.eta(), cl3d.phi(), cl3d.hOverE());
        if (scenario == UseEmInterp::EmOnly) // for emID objs, use EM interp as pT and set H = 0
        {
            if (isEM)
            {
                float pt_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
                float hoe_new = 0.;
                cluster = l1t::PFCluster(pt_new, cl3d.eta(), cl3d.phi(), hoe_new, isEM);
            }
        }
        else if (scenario == UseEmInterp::AllKeepHad) // for all objs, replace EM part with EM interp, preserve H
        {
            float had_old = cl3d.pt() - cluster.emEt();
            float em_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
            float pt_new = had_old + em_new;
            float hoe_new = em_new > 0 ? (had_old / em_new) : -1;
            cluster = l1t::PFCluster(pt_new, cl3d.eta(), cl3d.phi(), hoe_new, isEM);
        }
        else if (scenario == UseEmInterp::AllKeepTot) // for all objs, replace EM part with EM interp, preserve pT
        {
            float em_new = cl3d.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
            float hoe_new = em_new > 0 ? (cl3d.pt() / em_new - 1) : -1;
            cluster = l1t::PFCluster(cl3d.pt(), cl3d.eta(), cl3d.phi(), hoe_new, isEM);
        }

        if (!VsPuId.method().empty())
        { 
            int id = VsPuId.passID(*cl3dIt, cluster);
            // if (!id) { continue; } // skip if it does not pass puid
            HGCluster.puId = id;
            HGCluster.puIdScore = cluster.egVsPUMVAOut();
        }
        if (!VsPionId.method().empty())
        {
            HGCluster.pionId = VsPionId.passID(*cl3dIt, cluster);
            HGCluster.pionIdScore = cluster.egVsPionMVAOut();
        }
        // END PU/PION ID

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