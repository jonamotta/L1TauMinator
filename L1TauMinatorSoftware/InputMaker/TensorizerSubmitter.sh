source /opt/exp_soft/cms/t3/t3setup

python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 9x9 --doTens4Calib --doTens4Ident --doTestRun
python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 7x7 --doTens4Calib --doTens4Ident --doTestRun
python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 5x5 --doTens4Calib --doTens4Ident --doTestRun
python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 5x9 --doTens4Calib --doTens4Ident --doTestRun