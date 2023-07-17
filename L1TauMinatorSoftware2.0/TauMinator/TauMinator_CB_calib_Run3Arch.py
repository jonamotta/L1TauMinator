from tensorflow.keras.initializers import RandomNormal as RN
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cmsml
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def inspectWeights(model, which):
    if which=='kernel': idx=0
    if which=='bias':   idx=1

    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[idx].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10,10))
    bins = np.linspace(-0.4, 0.4, 50)
    plt.hist(histosW,bins,histtype='step',stacked=True,label=labelsW)
    plt.legend(frameon=False,loc='upper left', fontsize=16)
    plt.ylabel('Recurrence')
    plt.xlabel('Weight value')
    plt.xlim(-0.7,0.5)
    plt.yscale('log')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/modelSparsity'+which+'.pdf')
    plt.close()

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def make_AddList(TTP, inputs, name=""):
    AdditionList = []
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,0,:], name=f"{name}tt{0}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,1,:], name=f"{name}tt{1}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,2,:], name=f"{name}tt{2}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,3,:], name=f"{name}tt{3}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,4,:], name=f"{name}tt{4}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,5,:], name=f"{name}tt{5}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,6,:], name=f"{name}tt{6}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,7,:], name=f"{name}tt{7}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,0,8,:], name=f"{name}tt{8}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,0,:], name=f"{name}tt{9}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,1,:], name=f"{name}tt{10}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,2,:], name=f"{name}tt{11}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,3,:], name=f"{name}tt{12}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,4,:], name=f"{name}tt{13}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,5,:], name=f"{name}tt{14}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,6,:], name=f"{name}tt{15}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,7,:], name=f"{name}tt{16}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,1,8,:], name=f"{name}tt{17}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,0,:], name=f"{name}tt{18}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,1,:], name=f"{name}tt{19}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,2,:], name=f"{name}tt{20}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,3,:], name=f"{name}tt{21}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,4,:], name=f"{name}tt{22}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,5,:], name=f"{name}tt{23}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,6,:], name=f"{name}tt{24}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,7,:], name=f"{name}tt{25}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,2,8,:], name=f"{name}tt{26}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,0,:], name=f"{name}tt{27}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,1,:], name=f"{name}tt{28}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,2,:], name=f"{name}tt{29}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,3,:], name=f"{name}tt{30}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,4,:], name=f"{name}tt{31}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,5,:], name=f"{name}tt{32}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,6,:], name=f"{name}tt{33}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,7,:], name=f"{name}tt{34}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,3,8,:], name=f"{name}tt{35}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,0,:], name=f"{name}tt{36}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,1,:], name=f"{name}tt{37}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,2,:], name=f"{name}tt{38}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,3,:], name=f"{name}tt{39}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,4,:], name=f"{name}tt{40}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,5,:], name=f"{name}tt{41}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,6,:], name=f"{name}tt{42}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,7,:], name=f"{name}tt{43}")(inputs)) )
    AdditionList.append( TTP(layers.Lambda(lambda x : x[:,4,8,:], name=f"{name}tt{44}")(inputs)) )
    return AdditionList

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                default=None)
    parser.add_option("--date",         dest="date",                             default=None)
    parser.add_option("--inTag",        dest="inTag",                            default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
    parser.add_option('--train',        dest='train',       action='store_true', default=False)
    parser.add_option('--dm_weighted',  dest='dm_weighted', action='store_true', default=False)
    parser.add_option('--pt_weighted',  dest='pt_weighted', action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training'+options.inTag
    tag = ''
    if options.dm_weighted: tag = '_dmWeighted'
    if options.pt_weighted: tag = '_ptWeighted'
    outdir += tag
    os.system('mkdir -p '+outdir+'/TauMinator_CB_calib_plots')
    os.system('mkdir -p '+indir+'/CMSSWmodels'+tag)

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(outdir+'/tensors/images_train.npz')['arr_0']
    X2 = np.load(outdir+'/tensors/posits_train.npz')['arr_0']
    Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
    Yid  = Y[:,1].reshape(-1,1)
    Ycal = Y[:,0].reshape(-1,1)

    # select only taus
    tau_sel = Yid.reshape(1,-1)[0] > 0
    X1 = X1[tau_sel]
    X2 = X2[tau_sel]
    Y = Y[tau_sel]
    Ycal = Ycal[tau_sel]

    # FIXME #
    pt_sel = (Ycal.reshape(1,-1)[0]>40) & (Ycal.reshape(1,-1)[0]<60)
    X1 = X1[pt_sel]
    X2 = X2[pt_sel]
    Y = Y[pt_sel]
    Ycal = Ycal[pt_sel]

    if options.dm_weighted:
        # create dm weights
        Ydm = Y[:,4].reshape(-1,1)
        dm0 = Ydm.reshape(1,-1)[0] == 0
        dm1 = (Ydm.reshape(1,-1)[0] == 1) | (Ydm.reshape(1,-1)[0] == 2)
        dm10 = Ydm.reshape(1,-1)[0] == 10
        dm11 = (Ydm.reshape(1,-1)[0] == 11) | (Ydm.reshape(1,-1)[0] == 12)

        dm0_w = np.sum(tau_sel) / np.sum(dm0)
        dm1_w = np.sum(tau_sel) / np.sum(dm1)
        dm10_w = np.sum(tau_sel) / np.sum(dm10)
        dm11_w = np.sum(tau_sel) / np.sum(dm11)

        dm_weights = dm0_w*dm0 + dm1_w*dm1 + dm10_w*dm10 + dm11_w*dm11

    # if options.pt_weighted:
    def customPtDownWeight(row):
        if (row['gen_pt']<20):
            return row['weight']*0.5

        elif ((row['gen_pt']>100)&(row['gen_pt']<150)):
            return row['weight']*0.5
        
        elif (row['gen_pt']>150):
            return 1.0

        else:
            return row['weight']

    dfweights = pd.DataFrame(Y[:,0], columns=['gen_pt'])
    weight_Ebins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 130, 150, 2000]
    dfweights['gen_pt_bin'] = pd.cut(dfweights['gen_pt'], bins=weight_Ebins, labels=False, include_lowest=True)
    dfweights['weight'] = dfweights.shape[0] / dfweights.groupby(['gen_pt_bin'])['gen_pt_bin'].transform('count')
    dfweights['weight'] = dfweights.apply(lambda row: customPtDownWeight(row) , axis=1)
    pt_weights = dfweights['weight'].to_numpy()

    plt.figure(figsize=(10,10))
    plt.hist(dfweights['gen_pt'], bins=weight_Ebins,                               label="Un-weighted", color='red',   lw=2, histtype='step', alpha=0.7)
    plt.hist(dfweights['gen_pt'], bins=weight_Ebins, weights=dfweights['weight'],  label="Weighted",    color='green', lw=2, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.yscale('log')
    plt.xlim(0,175)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/pt_gentau.pdf')
    plt.close()

    del dfweights

    # print(dfweights.groupby('gen_pt_bin')['weight'].mean())
    # exit()


    ############################## Models definition ##############################

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauMinator_CB_calib_plots/training_calib.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')
    
        layer1 = layers.Dense(32, name='nn1', input_dim=3, activation='relu', kernel_initializer=RN(seed=7), use_bias=False)
        layer2 = layers.Dense(64, name='nn2',              activation='relu', kernel_initializer=RN(seed=7), use_bias=False)
        layer2 = layers.Dense(64, name='nn3',              activation='relu', kernel_initializer=RN(seed=7), use_bias=False)
        layer2 = layers.Dense(32, name='nn4',              activation='relu', kernel_initializer=RN(seed=7), use_bias=False)
        layer3 = layers.Dense(1,  name='nn5',              activation='linear', kernel_initializer=RN(seed=7), use_bias=False)
        # layer4 = lay.Lambda(Fgrad)

        TTP = tf.keras.models.Sequential(name="ttp")
        TTP.add(layer1)
        TTP.add(layer2)
        TTP.add(layer3)
        # TTP.add(layer4)
        PredictionList = make_AddList(TTP, images)
        output = layers.Add(name="calibratedPt")(PredictionList)

        TauMinatorModel = keras.Model(inputs=images, outputs=output, name='TauMinator_CB_calib')

        def custom_loss(y_true, y_pred):
            return tf.nn.l2_loss((y_true - y_pred)/(y_true+0.1))

        TauMinatorModel.compile(optimizer=keras.optimizers.Nadam(), #learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                loss=custom_loss,
                                metrics=['RootMeanSquaredError'],
                                # sample_weight_mode='sample-wise',
                                run_eagerly=True)


        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauMinatorModel.fit(X1, Ycal, epochs=35, batch_size=1024, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks)#, sample_weight=pt_weights)

        TauMinatorModel.save(outdir+'/CAL_DNNmodel', include_optimizer=False)
        # cmsml.tensorflow.save_graph(indir+'/CMSSWmodels'+tag+'/DNNcalib_CB.pb', TauMinatorModel, variables_to_constants=True)
        # cmsml.tensorflow.save_graph(indir+'/CMSSWmodels'+tag+'/DNNcalib_CB.pb.txt', TauMinatorModel, variables_to_constants=True)

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CB_calib_plots/'+metric+'.pdf')
                plt.close()

            else:
                if 'val_' in metric: continue

                plt.plot(history.history[metric], label='Training dataset', lw=2)
                plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
                plt.xlabel('Epoch')
                if 'loss' in metric:
                    plt.ylabel('Loss')
                    plt.legend(loc='upper right')
                elif 'auc' in metric:
                    plt.ylabel('AUC')
                    plt.legend(loc='lower right')
                elif 'binary_accuracy' in metric:
                    plt.ylabel('Binary accuracy')
                    plt.legend(loc='lower right')
                elif 'root_mean_squared_error' in metric:
                    plt.ylabel('Root Mean Squared Error')
                    plt.legend(loc='upper right')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CB_calib_plots/'+metric+'.pdf')
                plt.close()

        # restore normal output
        sys.stdout = sys.__stdout__

    else:
        TauMinatorModel = keras.models.load_model(outdir+'/CAL_DNNmodel', compile=False)

    ############################## Model validation and pots ##############################

    X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
    X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
    Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
    Yid_valid  = Y_valid[:,1].reshape(-1,1)
    Ycal_valid = Y_valid[:,0].reshape(-1,1)

    # select only taus
    tau_sel_valid = Yid_valid.reshape(1,-1)[0] > 0
    X1_valid = X1_valid[tau_sel_valid]
    X2_valid = X2_valid[tau_sel_valid]
    Y_valid = Y_valid[tau_sel_valid]
    Ycal_valid = Ycal_valid[tau_sel_valid]

    # FIXME #
    pt_sel = (Ycal_valid.reshape(1,-1)[0]>40) & (Ycal_valid.reshape(1,-1)[0]<60)
    X1_valid = X1_valid[pt_sel]
    X2_valid = X2_valid[pt_sel]
    Y_valid = Y_valid[pt_sel]
    Ycal_valid = Ycal_valid[pt_sel]

    train_calib = TauMinatorModel.predict(X1)
    valid_calib = TauMinatorModel.predict(X1_valid)

    ####### CALIB PART #######

    # dfLL = pd.DataFrame()
    # dfLL['calib_pt'] = train_calib.ravel()
    # dfLL['gen_pt']   = Ycal
    # dfLL['response'] = dfLL['calib_pt'] / dfLL['gen_pt']
    # # dfLL['gen_pt_binned']  = pd.cut(dfLL['gen_pt'],
    # #                                 bins=[18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500],
    # #                                 labels=False,
    # #                                 include_lowest=True)

    # dfLL['gen_pt_binned']  = ((dfLL['gen_pt'] - 18)/5).astype('int32')

    # meansTrainPt = dfLL.groupby('gen_pt_binned').mean()
    # meansTrainPt['logpt1'] = np.log(meansTrainPt['calib_pt'])
    # meansTrainPt['logpt2'] = meansTrainPt.logpt1**2
    # meansTrainPt['logpt3'] = meansTrainPt.logpt1**3
    # meansTrainPt['logpt4'] = meansTrainPt.logpt1**4

    # input_LL = meansTrainPt[['logpt1', 'logpt2', 'logpt3', 'logpt4']]
    # target_LL = meansTrainPt['response']
    # LogLinearizer = LinearRegression().fit(input_LL, target_LL)

    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel()
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,2].ravel()
    dfTrain['gen_phi']    = Y[:,3].ravel()
    # dfTrain['gen_dm']     = Y[:,4].ravel()
    
    # logpt1 = np.log(abs(dfTrain['calib_pt']))
    # logpt2 = logpt1**2
    # logpt3 = logpt1**3
    # logpt4 = logpt1**4
    # dfTrain['calib_pt_LL'] = dfTrain['calib_pt'] / LogLinearizer.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,2].ravel()
    dfValid['gen_phi']    = Y_valid[:,3].ravel()
    # dfValid['gen_dm']     = Y_valid[:,4].ravel()

    # logpt1 = np.log(abs(dfValid['calib_pt']))
    # logpt2 = logpt1**2
    # logpt3 = logpt1**3
    # logpt4 = logpt1**4
    # dfValid['calib_pt_LL'] = dfValid['calib_pt'] / LogLinearizer.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS ETA
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$|\eta^{Gen \tau}|$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_eta_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PHI
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$\phi^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_phi_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PT
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$p_{T}^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    # plt.xlim(-0.1,5)
    plt.xlim(0.0,2.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_pt_comparison.pdf')
    plt.close()


    # L1 TO GEN MAPPING
    # bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 250, 2000]
    bins = [40, 45, 50, 55, 60]
    dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'], bins=bins, labels=False, include_lowest=True)
    dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'], bins=bins, labels=False, include_lowest=True)

    # pt_bins_centers = np.append(np.arange(17.5,152.5,5), [200, 1125])
    pt_bins_centers = np.array(np.arange(42.5,58.5,5))

    trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].mean())
    validL1 = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].mean())
    trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].std())
    validL1std = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].std())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean())
    validL1 = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean())
    trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std())
    validL1std = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].std())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/GenToUncalinL1_pt.pdf')
    plt.close()

    scale_vs_pt_uncalib = []
    resol_vs_pt_uncalib = []
    scale_vs_pt_calib = []
    resol_vs_pt_calib = []

    for ibin in np.sort(dfValid['gen_pt_bin'].unique()):
        ledge = bins[ibin]
        uedge = bins[ibin+1]

        tmpTrain = dfTrain[dfTrain['gen_pt_bin']==ibin]
        tmpValid = dfValid[dfValid['gen_pt_bin']==ibin]

        plt.figure(figsize=(10,10))
        plt.hist(tmpValid['uncalib_pt']/tmpValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']), np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpTrain['calib_pt']/tmpTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpTrain['calib_pt']/tmpTrain['gen_pt']), np.std(tmpTrain['calib_pt']/tmpTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpValid['calib_pt']/tmpValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['calib_pt']/tmpValid['gen_pt']), np.std(tmpValid['calib_pt']/tmpValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
        plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
        plt.ylabel(r'a.u.')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison_'+str(ledge)+'pt'+str(uedge)+'.pdf')
        plt.close()

        scale_vs_pt_uncalib.append(np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_uncalib.append(np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

        scale_vs_pt_calib.append(np.mean(tmpValid['calib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_calib.append(np.std(tmpValid['calib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

    def expftc(x, A,B,C):
        return A*np.exp(-x*B)+C

    # scale vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, scale_vs_pt_uncalib, xerr=2.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_calib, xerr=2.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    p0 = [1,1,1]
    popt, pcov = curve_fit(expftc, pt_bins_centers, scale_vs_pt_calib, p0, maxfev=5000)
    plt.plot(np.linspace(1,150,150), expftc(np.linspace(1,150,150), *popt), '-', label='_', lw=1.5, color='green', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(0,150)
    plt.ylim(0.5,1.5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/scale_vs_pt.pdf')
    plt.close()

    # resolution vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, resol_vs_pt_uncalib, xerr=2.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_calib, xerr=2.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(0,150)
    plt.ylim(0.1,0.4)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/resolution_vs_pt.pdf')
    plt.close()




    dfTrain['calib_pt_corr'] = dfTrain['calib_pt'] / expftc(dfTrain['calib_pt'], *popt)
    dfValid['calib_pt_corr'] = dfValid['calib_pt'] / expftc(dfValid['calib_pt'], *popt)

    scale_vs_pt_uncalib = []
    resol_vs_pt_uncalib = []
    scale_vs_pt_calib = []
    resol_vs_pt_calib = []

    for ibin in np.sort(dfValid['gen_pt_bin'].unique()):
        ledge = bins[ibin]
        uedge = bins[ibin+1]

        tmpTrain = dfTrain[dfTrain['gen_pt_bin']==ibin]
        tmpValid = dfValid[dfValid['gen_pt_bin']==ibin]

        plt.figure(figsize=(10,10))
        plt.hist(tmpValid['uncalib_pt']/tmpValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']), np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpTrain['calib_pt_corr']/tmpTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpTrain['calib_pt_corr']/tmpTrain['gen_pt']), np.std(tmpTrain['calib_pt_corr']/tmpTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpValid['calib_pt_corr']/tmpValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['calib_pt_corr']/tmpValid['gen_pt']), np.std(tmpValid['calib_pt_corr']/tmpValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
        plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
        plt.ylabel(r'a.u.')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison_'+str(ledge)+'pt'+str(uedge)+'_corr.pdf')
        plt.close()

        scale_vs_pt_uncalib.append(np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_uncalib.append(np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

        scale_vs_pt_calib.append(np.mean(tmpValid['calib_pt_corr']/tmpValid['gen_pt']))
        resol_vs_pt_calib.append(np.std(tmpValid['calib_pt_corr']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

    # scale vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, scale_vs_pt_uncalib, xerr=2.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_calib, xerr=2.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(0,150)
    plt.ylim(0.5,1.5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/scale_vs_pt_corr.pdf')
    plt.close()

    # resolution vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, resol_vs_pt_uncalib, xerr=2.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_calib, xerr=2.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(0,150)
    plt.ylim(0.1,0.4)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/resolution_vs_pt_corr.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt_corr']/dfTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt_corr']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt_corr']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt_corr']/dfValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt_corr']/dfValid['gen_pt']), np.std(dfValid['calib_pt_corr']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison_corr.pdf')
    plt.close()

    print('*****************************************')
    print('Scale correction function parameters')
    print('    function = A * exp(-x*B) + C')
    print('        A =', popt[0])
    print('        B =', popt[1])
    print('        C =', popt[2])
    print('*****************************************')

















