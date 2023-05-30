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

#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterEgID.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TauMinator/DataFormats/interface/TowerHelper.h"
#include "L1TauMinator/DataFormats/interface/HGClusterHelper.h"
#include "L1TauMinator/DataFormats/interface/TauHelper.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"


class L1CaloTauProducer : public edm::stream::EDProducer<> {
    public:
        explicit L1CaloTauProducer(const edm::ParameterSet&);

    private:
        //----edm control---
        void produce(edm::Event&, const edm::EventSetup&) override;

        //----private functions----
        int tower_dIPhi(int &iPhi_1, int &iPhi_2) const;
        int tower_dIEta(int &iEta_1, int &iEta_2) const;
        int endcap_iphi(float &phi) const;
        int endcap_ieta(float &eta) const;
        float inputQuantizer(float inputF, float LSB, int nbits);

        //----tokens and handles----
        edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowersToken;
        edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

        edm::EDGetToken hgcalTowersToken;
        edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowersHandle;

        edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> HGClusterToken;
        edm::Handle<l1t::HGCalMulticlusterBxCollection> HGClusterHandle;

        //----private variables----
        enum class UseEmInterp { No, EmOnly, AllKeepHad, AllKeepTot };
        UseEmInterp scenario;
        StringCutObjectSelector<l1t::HGCalMulticluster> preEmId;
        l1tpf::HGC3DClusterEgID VsPuId;

        double EcalEtMinForClustering;
        double HcalEtMinForClustering;
        double EtMinForSeeding;

        std::string CNNmodel_CB_path;
        std::string DNNident_CB_path;
        std::string DNNcalib_CB_path;

        std::string CNNmodel_CE_path;
        std::string DNNident_CE_path;
        std::string DNNcalib_CE_path;

        bool DEBUG;

        tensorflow::GraphDef* CNNmodel_CB;
        tensorflow::GraphDef* DNNident_CB;
        tensorflow::GraphDef* DNNcalib_CB;

        tensorflow::Session* CNNmodel_CBsession;
        tensorflow::Session* DNNident_CBsession;
        tensorflow::Session* DNNcalib_CBsession;

        tensorflow::GraphDef* CNNmodel_CE;
        tensorflow::GraphDef* DNNident_CE;
        tensorflow::GraphDef* DNNcalib_CE;

        tensorflow::Session* CNNmodel_CEsession;
        tensorflow::Session* DNNident_CEsession;
        tensorflow::Session* DNNcalib_CEsession;

        // hardoced dimensions of the tower clusters
        int seedIdx = 22;
        int IEta_dim = 5;
        int IPhi_dim = 9;
        float Eta_dim = 0.2;
        float Phi_dim = 0.4;
        float Eta_dim_seed = 0.35;
        float Phi_dim_seed = 0.7;
};


/*
██ ███    ███ ██████  ██      ███████ ███    ███ ███████ ███    ██ ████████  █████  ████████ ██  ██████  ███    ██
██ ████  ████ ██   ██ ██      ██      ████  ████ ██      ████   ██    ██    ██   ██    ██    ██ ██    ██ ████   ██
██ ██ ████ ██ ██████  ██      █████   ██ ████ ██ █████   ██ ██  ██    ██    ███████    ██    ██ ██    ██ ██ ██  ██
██ ██  ██  ██ ██      ██      ██      ██  ██  ██ ██      ██  ██ ██    ██    ██   ██    ██    ██ ██    ██ ██  ██ ██
██ ██      ██ ██      ███████ ███████ ██      ██ ███████ ██   ████    ██    ██   ██    ██    ██  ██████  ██   ████
*/

// ----Constructor and Destructor -----
L1CaloTauProducer::L1CaloTauProducer(const edm::ParameterSet& iConfig) 
    : l1TowersToken(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers"))),
      hgcalTowersToken(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("hgcalTowers"))),

      HGClusterToken(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("HgcalClusters"))),
      scenario(UseEmInterp::No),
      preEmId(iConfig.getParameter<std::string>("preEmId")),
      VsPuId(iConfig.getParameter<edm::ParameterSet>("VsPuId")),
      
      EcalEtMinForClustering(iConfig.getParameter<double>("EcalEtMinForClustering")),
      HcalEtMinForClustering(iConfig.getParameter<double>("HcalEtMinForClustering")),
      EtMinForSeeding(iConfig.getParameter<double>("EtMinForSeeding")),
      
      CNNmodel_CB_path(iConfig.getParameter<std::string>("CNNmodel_CB_path")),
      DNNident_CB_path(iConfig.getParameter<std::string>("DNNident_CB_path")),
      DNNcalib_CB_path(iConfig.getParameter<std::string>("DNNcalib_CB_path")),
      CNNmodel_CE_path(iConfig.getParameter<std::string>("CNNmodel_CE_path")),
      DNNident_CE_path(iConfig.getParameter<std::string>("DNNident_CE_path")),
      DNNcalib_CE_path(iConfig.getParameter<std::string>("DNNcalib_CE_path")),

      DEBUG(iConfig.getParameter<bool>("DEBUG"))
{    

    // Create sessions for Tensorflow inferece
    CNNmodel_CB = tensorflow::loadGraphDef(CNNmodel_CB_path);
    CNNmodel_CBsession = tensorflow::createSession(CNNmodel_CB);

    DNNident_CB = tensorflow::loadGraphDef(DNNident_CB_path);
    DNNident_CBsession = tensorflow::createSession(DNNident_CB);

    DNNcalib_CB = tensorflow::loadGraphDef(DNNcalib_CB_path);
    DNNcalib_CBsession = tensorflow::createSession(DNNcalib_CB);

    CNNmodel_CE = tensorflow::loadGraphDef(CNNmodel_CE_path);
    CNNmodel_CEsession = tensorflow::createSession(CNNmodel_CE);

    DNNident_CE = tensorflow::loadGraphDef(DNNident_CE_path);
    DNNident_CEsession = tensorflow::createSession(DNNident_CE);

    DNNcalib_CE = tensorflow::loadGraphDef(DNNcalib_CE_path);
    DNNcalib_CEsession = tensorflow::createSession(DNNcalib_CE);

    // Initialize HGCAL BDTs
    if (!VsPuId.method().empty()) {
      VsPuId.prepareTMVA();
    }

    // Create produced outputs
    produces<TowerHelper::SimpleTowerClustersCollection> ("l1TowerClustersNxMCB");
    produces<TowerHelper::SimpleTowerClustersCollection> ("l1TowerClustersNxMCE");
    produces<HGClusterHelper::HGClustersCollection>      ("HGClustersCollection");
    produces<TauHelper::TausCollection>                  ("TauMinatorTausCollection");

    std::cout << "EtMinForSeeding = " << EtMinForSeeding << " , HcalTpEtMin = " << HcalEtMinForClustering << " , EcalTpEtMin = " << EcalEtMinForClustering << std::endl;
}

void L1CaloTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& eSetup)
{
    // Create and Fill collection of all calotowers and their attributes
    std::vector<TowerHelper::SimpleTowerHit> l1CaloTowers;

    iEvent.getByToken(l1TowersToken, l1CaloTowerHandle);
    int warnings = 0;
    for (auto &hit : *l1CaloTowerHandle.product())
    {
        // skip this weird towers and store warning
        if (hit.towerIEta() == -1016 && hit.towerIPhi() == -962)
        {
            warnings += 1;
            continue;
        }

        TowerHelper::SimpleTowerHit l1Hit;
        l1Hit.isBarrel     = true;
        l1Hit.l1egTowerEt  = hit.l1egTowerEt();
        l1Hit.nL1eg        = hit.nL1eg();
        l1Hit.towerEta     = hit.towerEta();
        l1Hit.towerPhi     = hit.towerPhi();
        l1Hit.towerEm      = hit.ecalTowerEt();
        l1Hit.towerHad     = hit.hcalTowerEt();
        l1Hit.towerEt      = l1Hit.towerEm + l1Hit.towerHad + l1Hit.l1egTowerEt;
        l1Hit.towerIeta    = hit.towerIEta();
        l1Hit.towerIphi    = hit.towerIPhi();

        l1CaloTowers.push_back(l1Hit);
    }
    if (warnings != 0) { std::cout << " ** WARNING : FOUND " << warnings << " TOWERS WITH towerIeta=-1016 AND towerIphi=-962" << std::endl; }

    iEvent.getByToken(hgcalTowersToken, hgcalTowersHandle);
    for (auto &hit : *hgcalTowersHandle.product())
    {
        TowerHelper::SimpleTowerHit l1Hit;
        l1Hit.isBarrel     = false;
        l1Hit.l1egTowerEt  = 0.0;
        l1Hit.nL1eg        = 0;
        l1Hit.towerEta     = hit.eta();
        l1Hit.towerPhi     = hit.phi();
        l1Hit.towerEm      = hit.etEm();
        l1Hit.towerHad     = hit.etHad();
        l1Hit.towerEt      = l1Hit.towerEm + l1Hit.towerHad;
        l1Hit.towerIeta    = endcap_ieta(l1Hit.towerEta);
        l1Hit.towerIphi    = endcap_iphi(l1Hit.towerPhi);

        l1CaloTowers.push_back(l1Hit);
    }

    // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
    std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const TowerHelper::SimpleTowerHit &a, TowerHelper::SimpleTowerHit &b) { return a.towerEt > b.towerEt; });

    // Create and Fill the collection of 3D clusters and their attributes
    std::vector<HGClusterHelper::HGCluster> AllHGClustersCollection;
    iEvent.getByToken(HGClusterToken, HGClusterHandle);

    for (auto cl3dIt = HGClusterHandle->begin(0); cl3dIt != HGClusterHandle->end(0); ++cl3dIt)
    {
        auto& cl3d = *cl3dIt;

        // Implement cl3d PU ID as done in https://github.com/cms-sw/cmssw/blob/master/L1Trigger/Phase2L1ParticleFlow/plugins/PFClusterProducerFromHGC3DClusters.cc#L120
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
            if (!id) { continue; } // skip cl3d if it does not pass puid
        }

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

        AllHGClustersCollection.push_back(HGCluster);
    }

    // order the collection in pt (the input to the GCT will be pt ordered)
    std::sort(begin(AllHGClustersCollection), end(AllHGClustersCollection), [](const HGClusterHelper::HGCluster &a, HGClusterHelper::HGCluster &b) { return a.pt > b.pt; });

    // Make NxM TowerClusters and HGClusters collections for TauMinator
    std::unique_ptr<TowerHelper::SimpleTowerClustersCollection> l1TowerClustersNxM_CB(new TowerHelper::SimpleTowerClustersCollection);
    std::unique_ptr<TowerHelper::SimpleTowerClustersCollection> l1TowerClustersNxM_CE(new TowerHelper::SimpleTowerClustersCollection);
    std::unique_ptr<HGClusterHelper::HGClustersCollection> HGClustersCollection(new HGClusterHelper::HGClustersCollection);

    // supporting collection of endcap clusters before cl3d matching
    std::vector<TowerHelper::SimpleTowerCluster> AllL1TowerClustersNxM_CE;

    bool caloTauSeedingFinished = false;
    // loop for seeding of different tau objects
    while (!caloTauSeedingFinished)
    {
        TowerHelper::SimpleTowerCluster clNxM; clNxM.InitHits(IEta_dim, IPhi_dim);
        bool seeded = false;

        // loop over towers to cluster
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip seeding in towers that would make the cluster extend in HF
            // skip l1CaloTowers which are already used by this clusters' mask
            if (abs(l1CaloTower.towerEta) > 2.83 || l1CaloTower.stale4seed) { continue; }

            // if not seded do the seeding
            if (!seeded)
            {
                // the leading unused tower has ET < min, stop jet clustering
                if (l1CaloTower.towerEt < EtMinForSeeding)
                {
                    caloTauSeedingFinished = true;
                    continue;
                }

                clNxM.seedIeta = l1CaloTower.towerIeta;
                clNxM.seedIphi = l1CaloTower.towerIphi;
                clNxM.seedEta  = l1CaloTower.towerEta;
                clNxM.seedPhi  = l1CaloTower.towerPhi;
                if (l1CaloTower.isBarrel) { clNxM.barrelSeeded = true; }

                clNxM.towerHits[seedIdx] = l1CaloTower;
                l1CaloTower.stale4seed = true;
                l1CaloTower.stale = true;
                seeded = true;

                continue;
            }

            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            // use iEta/iPhi comparisons in the barrel and eta/phi in HGCal
            if (clNxM.barrelSeeded && l1CaloTower.isBarrel)
            {
                d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM.seedIeta);
                d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM.seedIphi);
            }
            else
            {
                d_Eta = l1CaloTower.towerEta - clNxM.seedEta;
                d_Phi = reco::deltaPhi(l1CaloTower.towerPhi, clNxM.seedPhi);
            }

            // stale tower for seeding if it would lead to overalp between clusters
            if ((abs(d_iEta) <= IEta_dim-1 && abs(d_iPhi) <= IPhi_dim-1) || (abs(d_Eta) < Eta_dim_seed && abs(d_Phi) < Phi_dim_seed)) { l1CaloTower.stale4seed = true; }

        } // end for loop over TPs

        // pushback seeds split in barrel and endcap
        if (seeded)
        {
            if (clNxM.barrelSeeded) { l1TowerClustersNxM_CB->push_back(clNxM); }
            else                     { AllL1TowerClustersNxM_CE.push_back(clNxM); }
        }

    } // end while loop of TowerClusters seeding


    // loop for barrel NxM TowerClusters clustering starting from the seeds just found
    for (auto& clNxM : *l1TowerClustersNxM_CB)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used
            if (l1CaloTower.stale) { continue; }

            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            int   hitIdx = 99.;
            // use iEta/iPhi comparisons in the barrel and use eta/phi in HGCal
            if (l1CaloTower.isBarrel)
            {
                d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM.seedIeta);
                d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM.seedIphi);

                hitIdx = d_iEta * IPhi_dim + d_iPhi + seedIdx;
            }
            else
            {
                d_Eta = l1CaloTower.towerEta - clNxM.seedEta;
                d_Phi = reco::deltaPhi(l1CaloTower.towerPhi, clNxM.seedPhi);

                int dieta = d_Eta / 0.0807; // minimal difference in endcap is 0.0808
                int diphi = d_Phi / 0.0872; 
                hitIdx = dieta * IPhi_dim + diphi + seedIdx;
            }

            // cluster all towers in a NxM towers mask
            if ((abs(d_iEta) <= (IEta_dim-1)/2 && abs(d_iPhi) <= (IPhi_dim-1)/2) || (abs(d_Eta) < Eta_dim && abs(d_Phi) < Phi_dim))
            {
                clNxM.towerHits[hitIdx] = l1CaloTower;
                l1CaloTower.stale = true;
            }

        } // end for loop over TPs

    } // end while loop of barrel TowerClusters creation


    // in the endcap cross-loop over clNxM and cl3d to match them (we can do it before full clustering just using the seed info)
    for (auto& clNxM : AllL1TowerClustersNxM_CE)
    {
        bool matched = false;
        for (auto &HGCluster : AllHGClustersCollection)
        {
            // in case the clNxM or HGCluster have already been matched just continue through the list to the end
            // only use cl3ds above 4GeV
            if (matched || HGCluster.stale || HGCluster.pt < 4) { continue; }

            float d_Eta = HGCluster.eta - clNxM.seedEta;
            float d_Phi = reco::deltaPhi(HGCluster.phi, clNxM.seedPhi);
            float d_R2 = pow(d_Eta,2) + pow(d_Phi,2);

            if (d_R2 < 0.25)
            {
                HGCluster.stale = true;
                HGClustersCollection->push_back(HGCluster);
                l1TowerClustersNxM_CE->push_back(clNxM);
                matched = true;
            }
        
        } // end for loop over cl3ds
    
    } // end for loop over clNxM


    // loop for endcap matched NxM TowerClusters clustering starting from the seeds just found
    for (auto& clNxM : *l1TowerClustersNxM_CE)
    {
        for (auto &l1CaloTower : l1CaloTowers)
        {
            // skip l1CaloTowers which are already used
            if (l1CaloTower.stale) { continue; }

            int   d_iEta = 99;
            int   d_iPhi = 99;
            float d_Eta = 99.;
            float d_Phi = 99.;
            int   hitIdx = 99.;
            // use iEta/iPhi comparisons in the endcap and use eta/phi in HGCal
            if (l1CaloTower.isBarrel)
            {
                d_iEta = tower_dIEta(l1CaloTower.towerIeta, clNxM.seedIeta);
                d_iPhi = tower_dIPhi(l1CaloTower.towerIphi, clNxM.seedIphi);

                hitIdx = d_iEta * IPhi_dim + d_iPhi + seedIdx;
            }
            else
            {
                d_Eta = l1CaloTower.towerEta - clNxM.seedEta;
                d_Phi = reco::deltaPhi(l1CaloTower.towerPhi, clNxM.seedPhi);

                int dieta = d_Eta / 0.0807; // minimal difference in endcap is 0.0808
                int diphi = d_Phi / 0.0872; 
                hitIdx = dieta * IPhi_dim + diphi + seedIdx;
            }

            // cluster all towers in a NxM towers mask
            if ((abs(d_iEta) <= (IEta_dim-1)/2 && abs(d_iPhi) <= (IPhi_dim-1)/2) || (abs(d_Eta) < Eta_dim && abs(d_Phi) < Phi_dim))
            {
                clNxM.towerHits[hitIdx] = l1CaloTower;
                l1CaloTower.stale = true;
            }

        } // end for loop over TPs

    } // end while loop of endcap TowerClusters creation


    // Barrel TauMinator application
    tensorflow::setLogging("2");
    int batchSize_CB =  (int)(l1TowerClustersNxM_CB->size());
    tensorflow::TensorShape imageShape_CB({batchSize_CB, IEta_dim, IPhi_dim, 3});
    tensorflow::TensorShape positionShape_CB({batchSize_CB, 2});
    tensorflow::Tensor TowerClusterImage_CB(tensorflow::DT_FLOAT, imageShape_CB);
    tensorflow::Tensor TowerClusterPosition_CB(tensorflow::DT_FLOAT, positionShape_CB);

    int clIdx = 0;
    for (auto& clNxM : *l1TowerClustersNxM_CB)
    {
        // Fill inputs for Tensorflow inference
        for (int eta = 0; eta < IEta_dim; ++eta)
        {
            for (int phi = 0; phi < IPhi_dim; ++phi)
            {
                int towerIdx = eta*IPhi_dim + phi;
                TowerClusterImage_CB.tensor<float, 4>()(clIdx, eta, phi, 0) = inputQuantizer(clNxM.towerHits[towerIdx].l1egTowerEt, 0.25, 10);
                TowerClusterImage_CB.tensor<float, 4>()(clIdx, eta, phi, 1) = inputQuantizer(clNxM.towerHits[towerIdx].towerEm,     0.25, 10);
                TowerClusterImage_CB.tensor<float, 4>()(clIdx, eta, phi, 2) = inputQuantizer(clNxM.towerHits[towerIdx].towerHad,    0.25, 10);

                if (DEBUG)
                {
                    std::cout << "(" << eta << "," << phi << ")[" << towerIdx << "]        " << clNxM.towerHits[towerIdx].l1egTowerEt << "    " << clNxM.towerHits[towerIdx].towerEm << "    " << clNxM.towerHits[towerIdx].towerHad << "\n" << std::endl;
                    if (phi==8) { std::cout << "" << std::endl; }
                }
            }
        }
        
        TowerClusterPosition_CB.tensor<float, 2>()(clIdx, 0) = clNxM.seedEta;
        TowerClusterPosition_CB.tensor<float, 2>()(clIdx, 1) = clNxM.seedPhi;

        clIdx++; // increase batch index
    }

    // Apply CNN model
    tensorflow::NamedTensorList CNNmodel_CBinputList = {{"TowerClusterImage", TowerClusterImage_CB}, {"TowerClusterPosition", TowerClusterPosition_CB}};
    std::vector<tensorflow::Tensor> CNNmodel_CBoutputs;
    tensorflow::run(CNNmodel_CBsession, CNNmodel_CBinputList, {"TauMinator_CB_conv/middleMan/concat"}, &CNNmodel_CBoutputs);
    tensorflow::NamedTensorList DNN_CBinputsList = {{"middleMan", CNNmodel_CBoutputs[0]}};

    // Apply DNN for identification
    std::vector<tensorflow::Tensor> DNN_CBoutputsIdent;
    tensorflow::run(DNNident_CBsession, DNN_CBinputsList, {"TauMinator_CB_ident/sigmoid_IDout/Sigmoid"}, &DNN_CBoutputsIdent);

    // Apply DNN for calibration
    std::vector<tensorflow::Tensor> DNN_CBoutputsCalib;
    tensorflow::run(DNNcalib_CBsession, DNN_CBinputsList, {"TauMinator_CB_calib/DNNout/MatMul"}, &DNN_CBoutputsCalib);

    // Fill TauMinator output variables of TowerClusters
    clIdx = 0;
    for (auto& clNxM : *l1TowerClustersNxM_CB)
    {
        clNxM.IDscore = DNN_CBoutputsIdent[0].matrix<float>()(0, clIdx);
        clNxM.calibPt = DNN_CBoutputsCalib[0].matrix<float>()(0, clIdx);
        clIdx++; // increase batch index
    }


    // Endcap TauMinator application
    int batchSize_CE =  (int)(l1TowerClustersNxM_CE->size());
    tensorflow::TensorShape imageShape_CE({batchSize_CE, IEta_dim, IPhi_dim, 3});
    tensorflow::TensorShape positionShape_CE({batchSize_CE, 2});
    tensorflow::TensorShape cl3dfeatShape_CE({batchSize_CE, 8});
    tensorflow::Tensor TowerClusterImage_CE(tensorflow::DT_FLOAT, imageShape_CE);
    tensorflow::Tensor TowerClusterPosition_CE(tensorflow::DT_FLOAT, positionShape_CE);
    tensorflow::Tensor Cl3dShapeFeatures_CE(tensorflow::DT_FLOAT, cl3dfeatShape_CE);

    clIdx = 0;
    for (auto& clNxM : *l1TowerClustersNxM_CE)
    {
        // indexing of cl3ds is the same as the one of clNxMs
        HGClusterHelper::HGCluster HGClu = HGClustersCollection->at(clIdx);

        // Fill inputs for Tensorflow inference
        for (int eta = 0; eta < IEta_dim; ++eta)
        {
            for (int phi = 0; phi < IPhi_dim; ++phi)
            {
                int towerIdx = eta*IPhi_dim + phi;
                TowerClusterImage_CE.tensor<float, 4>()(clIdx, eta, phi, 0) = inputQuantizer(clNxM.towerHits[towerIdx].l1egTowerEt, 0.25, 10);
                TowerClusterImage_CE.tensor<float, 4>()(clIdx, eta, phi, 1) = inputQuantizer(clNxM.towerHits[towerIdx].towerEm,     0.25, 10);
                TowerClusterImage_CE.tensor<float, 4>()(clIdx, eta, phi, 2) = inputQuantizer(clNxM.towerHits[towerIdx].towerHad,    0.25, 10);

                if (DEBUG)
                {
                    std::cout << "(" << eta << "," << phi << ")[" << towerIdx << "]        " << clNxM.towerHits[towerIdx].l1egTowerEt << "    " << clNxM.towerHits[towerIdx].towerEm << "    " << clNxM.towerHits[towerIdx].towerHad << "\n" << std::endl;
                    if (phi==8) { std::cout << "" << std::endl; }
                }
            }
        }
        
        TowerClusterPosition_CE.tensor<float, 2>()(clIdx, 0) = clNxM.seedEta;
        TowerClusterPosition_CE.tensor<float, 2>()(clIdx, 1) = clNxM.seedPhi;

        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 0) = (inputQuantizer(HGClu.pt, 0.25, 14)                     - 6.07127) / 8.09958;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 1) = (inputQuantizer(abs(HGClu.eta)-1.321, 0.004, 9)         - 1.43884) / 0.31367;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 2) = (HGClu.showerlength                                     - 31.2058) / 7.66842;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 3) = (HGClu.coreshowerlength                                 - 10.0995) / 2.73062;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 4) = (inputQuantizer(HGClu.spptot, 0.0000153, 16)            - 0.02386) / 0.01520;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 5) = (inputQuantizer(HGClu.szz, 0.00153, 16)                 - 19.5851) / 12.7077;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 6) = (inputQuantizer(HGClu.srrtot, 0.0000153, 16)            - 0.00606) / 0.00129;
        Cl3dShapeFeatures_CE.tensor<float, 2>()(clIdx, 7) = (inputQuantizer(10*(abs(HGClu.meanz)-321.05), 0.5, 12)  - 215.552) / 104.794;
    
        clIdx++; // increase batch index
    }

    // Apply CNN model
    tensorflow::NamedTensorList CNNmodel_CEinputList = {{"TowerClusterImage", TowerClusterImage_CE}, {"TowerClusterPosition", TowerClusterPosition_CE}, {"AssociatedCl3dFeatures", Cl3dShapeFeatures_CE}};
    std::vector<tensorflow::Tensor> CNNmodel_CEoutputs;
    tensorflow::run(CNNmodel_CEsession, CNNmodel_CEinputList, {"TauMinator_CE_conv/middleMan/concat"}, &CNNmodel_CEoutputs);
    tensorflow::NamedTensorList DNN_CEinputsList = {{"middleMan", CNNmodel_CEoutputs[0]}};

    // Apply DNN for identification
    std::vector<tensorflow::Tensor> DNN_CEoutputsIdent;
    tensorflow::run(DNNident_CEsession, DNN_CEinputsList, {"TauMinator_CE_ident/sigmoid_IDout/Sigmoid"}, &DNN_CEoutputsIdent);

    // Apply DNN for calibration
    std::vector<tensorflow::Tensor> DNN_CEoutputsCalib;
    tensorflow::run(DNNcalib_CEsession, DNN_CEinputsList, {"TauMinator_CE_calib/DNNout/MatMul"}, &DNN_CEoutputsCalib);

    // Fill TauMinator output variables of TowerClusters
    clIdx = 0;
    for (auto& clNxM : *l1TowerClustersNxM_CE)
    {
        clNxM.IDscore = DNN_CEoutputsIdent[0].matrix<float>()(0, clIdx);
        clNxM.calibPt = DNN_CEoutputsCalib[0].matrix<float>()(0, clIdx);
        clIdx++; // increase batch index
    }


    // Create and Fill the collection of L1 taus and their usefull attributes
    std::unique_ptr<TauHelper::TausCollection> TauMinatorTausCollection(new TauHelper::TausCollection);

    // definition of barrel and endcap scale corretcor parameters
    // float A_CB = 2.3442768459838006;  float A_CE = 1.9210322940301963;
    // float B_CB = 0.1058388788344820;  float B_CE = 0.0911722319870048;
    // float C_CB = 0.8827768598575636;  float C_CE = 0.8598476032538743;

    for (auto& clNxM : *l1TowerClustersNxM_CB)
    {
        if (clNxM.IDscore<0/*FIXME*/) { continue; }

        TauHelper::Tau Tau;
        // set tau information for the barrel area
        Tau.pt  = clNxM.calibPt; // / (A_CB*exp(-clNxM.calibPt*B_CB)+C_CB);
        Tau.eta = clNxM.seedEta;
        Tau.phi = clNxM.seedPhi;
        Tau.isBarrel = true;
        Tau.IDscore = clNxM.IDscore;

        TauMinatorTausCollection->push_back(Tau);
    }

    for (auto& clNxM : *l1TowerClustersNxM_CE)
    {
        if (clNxM.IDscore<0/*FIXME*/) { continue; }

        TauHelper::Tau Tau;
        // set tau information for the barrel area
        Tau.pt  = clNxM.calibPt; // / (A_CE*exp(-clNxM.calibPt*B_CE)+C_CE);
        Tau.eta = clNxM.seedEta;
        Tau.phi = clNxM.seedPhi;
        Tau.isBarrel = false;
        Tau.IDscore = clNxM.IDscore;

        TauMinatorTausCollection->push_back(Tau);
    }

    iEvent.put(std::move(l1TowerClustersNxM_CB),    "l1TowerClustersNxMCB");
    iEvent.put(std::move(l1TowerClustersNxM_CE),    "l1TowerClustersNxMCE");
    iEvent.put(std::move(HGClustersCollection),     "HGClustersCollection");
    iEvent.put(std::move(TauMinatorTausCollection), "TauMinatorTausCollection");

} // end of produce function

int L1CaloTauProducer::tower_dIPhi(int &iPhi_1, int &iPhi_2) const
{
    int PI = 36;
    int result = iPhi_1 - iPhi_2;
    if (result > PI)   { result -= 2 * PI; }
    if (result <= -PI) { result += 2 * PI; } 
    return result;
}

int L1CaloTauProducer::tower_dIEta(int &iEta_1, int &iEta_2) const
{
    if (iEta_1 * iEta_2 > 0) { return iEta_1 - iEta_2; }
    else
    {
        if (iEta_1>0) { return iEta_1 - iEta_2 - 1; }
        else          { return iEta_1 - iEta_2 + 1; }
    }
}

int L1CaloTauProducer::endcap_iphi(float &phi) const
{
    float phi_step = 0.0872664;
    if (phi > 0) { return floor(phi / phi_step) + 1;  }
    else         { return floor(phi / phi_step) + 73; }
}

int L1CaloTauProducer::endcap_ieta(float &eta) const
{
    float eta_step = 0.0845;
    return floor(abs(eta)/eta_step) * std::copysign(1,eta);
}

float L1CaloTauProducer::inputQuantizer(float inputF, float LSB, int nbits)
{
    return min( floor(inputF/LSB), float(pow(2,nbits)-1) ) * LSB;
}


DEFINE_FWK_MODULE(L1CaloTauProducer);