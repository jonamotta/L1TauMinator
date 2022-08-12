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