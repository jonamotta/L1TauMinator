from sklearn.preprocessing import StandardScaler
from optparse import OptionParser
import pandas as pd
import numpy as np
import pickle
import sys
import os

np.random.seed(7)

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    (options, args) = parser.parse_args()
    print(options)

    indirA = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTIdentifierTraining'+options.inTag
    indirB = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTCalibratorTraining'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauDNNOptimization'
    os.system('mkdir -p '+outdir)

    feats = ['cl3d_pt', 'cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ']
    df1 = pd.read_pickle(indirA+'/X_Ident_BDT_forIdentifier.pkl')[feats]
    df2 = pd.read_pickle(indirA+'/X_Ident_BDT_forEvaluator.pkl')[feats]
    df3 = pd.read_pickle(indirB+'/X_Calib_BDT_forCalibrator.pkl')[feats]
    df4 = pd.read_pickle(indirB+'/X_Calib_BDT_forEvaluator.pkl')[feats]

    df = pd.concat([df1, df2, df3, df4], axis=0)

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df), columns=feats)

    save_obj(scaler, outdir+'/dnn_features_scaler.pkl')

    with open(outdir+'/dnn_features_scaler.txt', 'w') as f:
        f.write("## feature \t mean \t std ##\n")
        for i, item in enumerate(feats):
            f.write(item+" - "+str(scaler.mean_[i])+" - "+str(scaler.scale_[i])+"\n")
