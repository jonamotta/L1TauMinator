![Alt text](TAUminator_logo.png)

# L1TauMinator

Package for the development of a ML-based Phase-2 Level-1 Tau Trigger algorithm

## Installation isntructions

```shell
cmsrel CMSSW_12_5_2_patch1
cd CMSSW_12_5_2_patch1/src
cmsenv
git cms-init
git cms-merge-topic -u cms-l1t-offline:l1t-phase2-v64

git cms-addpkg L1Trigger/L1THGCal

git clone git@github.com:jonamotta/L1TauMinator.git

scram b -j 12
```

## Package structure

The package has two sub-packages:
* the CMSSW-based sub-package: 
    * `Dataformats` containing the dataformats for the CMSSW producers
    * `L1TauMinatorEmuator` containing the CMSSW producers and analyzers that produce the input to the Python-based sub-package (don't get fooled by the name: this is not a firmware emulator!)
* the Python-based sub-package:
    * `L1TauMinatrSoftware` containing all the Neural Networks development

## The CMSSW-based sub-package

`L1TauMinatorEmuator` has the usual structure of a CMSSW package:
* `inputFiles` contains the list of files to be used as input to the Ntuple producers
* `plugins` contains the producers and analyzers:
    * the `*Handlers.cc` produce the singe objects, i.e. tower clusters, 3d clusters, and gen paticles. (The `GenHandler` has two versions: one with a custom handling of the taus and one with the same definition of the enu Team. The difference is quite big in terms of pure definition, but the effect on the development is marginal)
    * the `Ntuplizer.cc` takes all the `*Handlers.cc` inputs and produces Ntuples to be used for the trainings
* the `python` folder contains all the configuration files
* the `test` folder contains the `cmsRun` configuration and the code to submit condor jobs
* the `utilities` folder contains some code to do debugging and optimisation of some working points (a bit messy but quite useful)

The final CMSSW version of the TauMinator can be found in this [PR #42840](https://github.com/cms-sw/cmssw/pull/42840) where it is also implemented as a partial emulator (this time a real firmware emulator).

## The Python-based sub-package

`L1TauMinatrSoftware*` contains all the Neural Networks development and optimisation down to the `hls4ml` implementation.
The three versions correspond to doifferent stages of the development, the latest ois 3.0 .

`L1TauMinatrSoftware3.0` contains four differet folders:
* `inputMaker` to read the Ntuples produced by the `L1TauMinatorEmuator` package:
    * `Chain2Tensor*` is the family of codes that reads from ROOT `TChain` and produce Python-(more-)frinedly inputs and NN readable TensorFlow tensors
    * `*Merger` is the family of codes that takes the miriad of tensors produced with `Chain2Tensor*` and creates a single TensorFlow tensor
    * Sidenote: this part works well as long as the tensors are not too big, if tensors get to big a more stable and elegant solution are TFRecords. They were never needed for thsi work, but an example of their use can be found in my code for the Calibraton [here](https://github.com/jonamotta/CaloL1CalibrationProducer/blob/master/L1NtupleReader/batchMerger.py).
* `TauDisplayer` to make tau footprints (almost never used tbh, a bit cumbersome code)
* `TauMinator` with all the code for the training of the NNs/BDTs
    * `TauMinator_CB_*` are all the codes for the training, quantizitaion, and  pruning for the model for the Barrel section
    * `TauMinator_CE_*` are all the codes for the training, quantizitaion, and  pruning for the model for the Endcap section
    * The sequence of use for both families of codes is:
        * `*_ident.py` to tarin the convolutional and identification part
        * `*_calib.py` to train the calibration part based on the convolutional of the previous step
        * `*_QNTZD.py` are the smae thing but with pruning and quantization of the NNs
    * `TauMinator_CB.py` and `TauMinator_CE.py` are a (failed) test to make a single NN for identification and calibration
* `TauMinator_hls4ml` with the Jupyter notebooks for the HLS conversion and the FPGA resources usage estimate:

## Final notes

This package has been developped by Jona Motta as part of his PhD Thesis.

The TauMinator algorithm has been documented in three pubblications with different levels of precision:
* Jona Motta's PhD Thesis [Development of machine learning based τ trigger algorithms and search for Higgs boson pair production in the bbττ decay channel with the CMS detector at the LHC](https://cds.cern.ch/record/2881939?ln=en)
* CMS's Detector Perfomance Summary Note [Hadronic Tau Reconstruction in the CMS Phase-2 Level-1 Trigger using NNs with Calorimetric Information](https://cds.cern.ch/record/2868783?ln=en)
* Jona Motta's Proceeding of Science "Development and firmware implementation of a Machine Learning based hadronic Tau lepton Level-1 Trigger algorithm in CMS for the HL-LHC" [PoS(EPS-HEP2023)590](https://cds.cern.ch/record/2879312)

When mentioning this work please always remember to add the due Credits and attach the correct Reference for completeness.