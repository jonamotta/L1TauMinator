![Alt text](https://cernbox.cern.ch/index.php/s/Czwj0noP4pBJRQN)

# L1TauMinator

Package for the development of a ML-based Phase-2 Level-1 Tau Trigger algorithm

## Installation isntructions

```shell
cmsrel CMSSW_12_3_0_pre4
cd CMSSW_12_3_0_pre4/src
cmsenv
git cms-init
git cms-merge-topic -u cms-l1t-offline:l1t-phase2-v3.4.41

git cms-addpkg L1Trigger/L1THGCal

git clone git@github.com:jonamotta/L1TauMinator.git

scram b -j 12
```

## Package structure

The package has two sub-packages:
* the CMSSW-based sub-package: 
    * `Dataformats` containing the dataformats for the CMSSW producers
    * `L1TauMinatorEmuator` containing the actual CMSSW producers and analyzers
* the Python-based sub-package:
    * `L1TauMinatrSoftware` containing all the algorithm development

## The CMSSW-based sub-package

`L1TauMinatorEmuator` has the usual structure of a CMSSW package:
* `inputFiles` contains the list of files to be used as input to the Ntuple producers
* `plugins` contains the producers and analyzers:
    * the `*Handlers.cc` produce the singe objects
    * the `Ntuplizer.cc` takes all the `*Handlers.cc` inputs and produces Ntuples to be used for the trainings
    * the `L1CaloTau*.cc` produce the tau objects in the standard manner, applying the full algorithm 
* the `python` folder contains all the configuration files
* the `test` folder contains the `cmsRun` configuration and the code to submit condor jobs
* the `utilities` folder contains some code to do debugging

## The Python-based sub-package

`L1TauMinatrSoftware` contains four differet folders:
* `inputMaker` to read the Ntuples produced by the `L1TauMinatorEmuator` package
* `TauDisplayer` to make tau footprints
* `TauMinator` with all the code for the training of the NNs/BDTs
    * `CLTW*.py` code is all the NN training, quantizitaion, and  pruning for the model for tower clusters
    * `CL3D*.py` cose is all the NN/BDT training, quantizitaion, and  pruning for the model for hgcal clusters
* `TauMinator_hls4ml` with the Jupyter notebooks for the HLS conversion and the FPGA resources usage estimate