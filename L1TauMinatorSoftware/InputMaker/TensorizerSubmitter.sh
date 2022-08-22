source /opt/exp_soft/cms/t3/t3setup

# python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 9x9 --doTens4Calib --doTens4Ident --doTestRun
# python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 7x7 --doTens4Calib --doTens4Ident --doTestRun
# python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 5x5 --doTens4Calib --doTens4Ident --doTestRun
# python TensorizerSubmitter.py --v 0 --date 2022_08_17 --caloClNxM 5x9 --doTens4Calib --doTens4Ident --doTestRun

python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 9x9 --doVBFH --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 7x7 --doVBFH --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x5 --doVBFH --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x9 --doVBFH --doTens4Calib --doTens4Ident

python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 9x9 --doHH --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 7x7 --doHH --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x5 --doHH --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x9 --doHH --doTens4Calib --doTens4Ident

python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 9x9 --doZp500 --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 7x7 --doZp500 --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x5 --doZp500 --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x9 --doZp500 --doTens4Calib --doTens4Ident

python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 9x9 --doZp1500 --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 7x7 --doZp1500 --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x5 --doZp1500 --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x9 --doZp1500 --doTens4Calib --doTens4Ident

python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 9x9 --doQCD --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 7x7 --doQCD --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x5 --doQCD --doTens4Calib --doTens4Ident
python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x9 --doQCD --doTens4Calib --doTens4Ident

# python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 9x9 --doMinBias --doTens4Calib --doTens4Ident
# python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 7x7 --doMinBias --doTens4Calib --doTens4Ident
# python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x5 --doMinBias --doTens4Calib --doTens4Ident
# python Tree2TensorSubmitter.py --v 1 --date 2022_08_20 --caloClNxM 5x9 --doMinBias --doTens4Calib --doTens4Ident