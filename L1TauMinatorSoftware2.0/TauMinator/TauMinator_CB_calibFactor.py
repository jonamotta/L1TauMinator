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
    dfweights['weight'] = dfweights['weight'] / min(dfweights['weight'])
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

    # print(dfweights) #.groupby('gen_pt_bin')['weight'].mean())
    # exit()

    X1_totE = np.sum(X1, (1,2,3)).reshape(-1,1)
    Yfactor = Ycal / X1_totE
    
    farctor_sel = (Yfactor.reshape(1,-1)[0] < 2.4) & (Yfactor.reshape(1,-1)[0] > 0.4)
    X1 = X1[farctor_sel]
    X2 = X2[farctor_sel]
    Y = Y[farctor_sel]
    Ycal = Ycal[farctor_sel]
    Yfactor = Yfactor[farctor_sel]

    plt.figure(figsize=(10,10))
    plt.hist(Yfactor, bins=np.arange(0,20,0.2))
    # plt.xlabel(r'$p_{T}^{Gen \tau}$')
    # plt.ylabel(r'a.u.')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    plt.yscale('log')
    plt.xlim(0,6)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/Yfactor.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(Ycal, Yfactor)
    # plt.xlabel(r'$p_{T}^{Gen \tau}$')
    # plt.ylabel(r'a.u.')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # plt.yscale('log')
    plt.xlim(39,61)
    plt.ylim(0,6)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/Yfactor_vs_pt.pdf')
    plt.close()


    dfweights = pd.DataFrame(Yfactor, columns=['yfactor'])
    weight_Ebins = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    dfweights['yfactorbin'] = pd.cut(dfweights['yfactor'], bins=weight_Ebins, labels=False, include_lowest=True)
    dfweights['weight'] = dfweights.shape[0] / dfweights.groupby(['yfactorbin'])['yfactorbin'].transform('count')
    # dfweights['weight'] = dfweights.apply(lambda row: customPtDownWeight(row) , axis=1)
    dfweights['weight'] = dfweights['weight'] / min(dfweights['weight'])
    yfactor_weights = dfweights['weight'].to_numpy()

    plt.figure(figsize=(10,10))
    plt.hist(dfweights['yfactor'], bins=weight_Ebins,                               label="Un-weighted", color='red',   lw=2, histtype='step', alpha=0.7)
    plt.hist(dfweights['yfactor'], bins=weight_Ebins, weights=dfweights['weight'],  label="Weighted",    color='green', lw=2, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.yscale('log')
    plt.xlim(0,6)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/Yfactor_weighted.pdf')
    plt.close()

    del dfweights


    # print(X1_totE.shape)
    # print(Ycal.shape)
    # print(Yfactor.shape)
    # exit()

    ############################## Models definition ##############################

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauMinator_CB_calib_plots/training_calib.log')
        print(options)

        middleMan = keras.Input(shape=194, name='middleMan')

        x = layers.Dense(128, use_bias=False, name="DNNlayer1")(middleMan)
        x = layers.Activation('relu', name='RELU_DNNlayer1')(x)
        x = layers.Dense(256, use_bias=False, name="DNNlayer2")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer2')(x)
        x = layers.Dense(128, use_bias=False, name="DNNlayer3")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer3')(x)
        x = layers.Dense(64, use_bias=False, name="DNNlayer4")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer4')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout", activation='linear')(x)
        
        TauCalibrated = x

        TauMinatorModel = keras.Model(inputs=middleMan, outputs=TauCalibrated, name='TauMinator_CB_calib')

        def custom_loss(y_pred, y_true):
            return 100 * tf.math.log(tf.cosh(y_pred - y_true))

        if options.dm_weighted or options.pt_weighted:
            TauMinatorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.LogCosh(),
                                    metrics=['RootMeanSquaredError'],
                                    sample_weight_mode='sample-wise',
                                    run_eagerly=True)

        else:
            TauMinatorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.LogCosh(),
                                    metrics=['RootMeanSquaredError'],
                                    sample_weight_mode='sample-wise',
                                    run_eagerly=True)

        # print(TauMinatorModel.summary())
        # exit()

        ############################## Model training ##############################

        if not options.pt_weighted: CNN = keras.models.load_model(outdir+'/CNNmodel', compile=False)
        else:                       CNN = keras.models.load_model(outdir.replace('_ptWeighted', '')+'/CNNmodel', compile=False)
        CNNprediction = CNN([X1, X2])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        if options.dm_weighted:
            history = TauMinatorModel.fit(CNNprediction, Yfactor, epochs=500, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=dm_weights)

        elif options.pt_weighted:
            history = TauMinatorModel.fit(CNNprediction, Yfactor, epochs=500, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=pt_weights)

        else:
            history = TauMinatorModel.fit(CNNprediction, Yfactor, epochs=500, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=yfactor_weights)

        TauMinatorModel.save(outdir+'/CAL_DNNmodel', include_optimizer=False)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels'+tag+'/DNNcalib_CB.pb', TauMinatorModel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels'+tag+'/DNNcalib_CB.pb.txt', TauMinatorModel, variables_to_constants=True)

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
        if not options.pt_weighted: CNN = keras.models.load_model(outdir+'/CNNmodel', compile=False)
        else:                       CNN = keras.models.load_model(outdir.replace('_ptWeighted', '')+'/CNNmodel', compile=False)
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

    train_calib = np.sum(X1, (1,2,3)).reshape(-1,1) * TauMinatorModel.predict(CNN([X1, X2]))
    valid_calib = np.sum(X1_valid, (1,2,3)).reshape(-1,1) * TauMinatorModel.predict(CNN([X1_valid, X2_valid]))

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
    dfTrain['gen_dm']     = Y[:,4].ravel()
    
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
    dfValid['gen_dm']     = Y_valid[:,4].ravel()

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

    # PLOTS PER DM
    DMdict = {
            0  : r'$h^{\pm}$',
            1  : r'$h^{\pm}\pi^{0}$',
            10 : r'$h^{\pm}h^{\mp}h^{\pm}$',
            11 : r'$h^{\pm}h^{\mp}h^{\pm}\pi^{0}$',
        }

    tmp0 = dfValid[dfValid['gen_dm']==0]
    tmp1 = dfValid[(dfValid['gen_dm']==1) | (dfValid['gen_dm']==2)]
    tmp10 = dfValid[dfValid['gen_dm']==10]
    tmp11 = dfValid[(dfValid['gen_dm']==11) | (dfValid['gen_dm']==12)]
    plt.figure(figsize=(10,10))
    plt.hist(tmp0['uncalib_pt']/tmp0['gen_pt'],   bins=np.arange(0.05,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['uncalib_pt']/tmp0['gen_pt']), np.std(tmp0['uncalib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['uncalib_pt']/tmp1['gen_pt'],   bins=np.arange(0.05,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['uncalib_pt']/tmp1['gen_pt']), np.std(tmp1['uncalib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['uncalib_pt']/tmp10['gen_pt'], bins=np.arange(0.05,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['uncalib_pt']/tmp10['gen_pt']), np.std(tmp10['uncalib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['uncalib_pt']/tmp11['gen_pt'], bins=np.arange(0.05,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['uncalib_pt']/tmp11['gen_pt']), np.std(tmp11['uncalib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(tmp0['calib_pt']/tmp0['gen_pt'],   bins=np.arange(0.05,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['calib_pt']/tmp0['gen_pt']), np.std(tmp0['calib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['calib_pt']/tmp1['gen_pt'],   bins=np.arange(0.05,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['calib_pt']/tmp1['gen_pt']), np.std(tmp1['calib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['calib_pt']/tmp10['gen_pt'], bins=np.arange(0.05,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['calib_pt']/tmp10['gen_pt']), np.std(tmp10['calib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['calib_pt']/tmp11['gen_pt'], bins=np.arange(0.05,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['calib_pt']/tmp11['gen_pt']), np.std(tmp11['calib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/calibrated_DM_responses_comparison.pdf')
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

    scale_vs_pt_uncalib_valid = []
    resol_vs_pt_uncalib_valid = []
    scale_vs_pt_calib_valid = []
    resol_vs_pt_calib_valid = []

    scale_vs_pt_uncalib_train = []
    resol_vs_pt_uncalib_train = []
    scale_vs_pt_calib_train = []
    resol_vs_pt_calib_train = []

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

        scale_vs_pt_uncalib_valid.append(np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_uncalib_valid.append(np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

        scale_vs_pt_calib_valid.append(np.mean(tmpValid['calib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_calib_valid.append(np.std(tmpValid['calib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

        scale_vs_pt_uncalib_train.append(np.mean(tmpTrain['uncalib_pt']/tmpTrain['gen_pt']))
        resol_vs_pt_uncalib_train.append(np.std(tmpTrain['uncalib_pt']/tmpTrain['gen_pt'])/np.mean(tmpTrain['uncalib_pt']/tmpTrain['gen_pt']))

        scale_vs_pt_calib_train.append(np.mean(tmpTrain['calib_pt']/tmpTrain['gen_pt']))
        resol_vs_pt_calib_train.append(np.std(tmpTrain['calib_pt']/tmpTrain['gen_pt'])/np.mean(tmpTrain['uncalib_pt']/tmpTrain['gen_pt']))

    def expftc(x, A,B,C):
        return A*np.exp(-x*B)+C

    # scale vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, scale_vs_pt_uncalib_valid, xerr=2.5, label=r'Uncalibrated Valid.',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_calib_valid, xerr=2.5,   label=r'Calibrated Valid.',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_uncalib_train, xerr=2.5, label=r'Uncalibrated Train.',  color='orange',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_calib_train, xerr=2.5,   label=r'Calibrated Train.',    color='blue', lw=2, alpha=0.7, marker='o', ls='None')
    p0 = [1,1,1]
    popt, pcov = curve_fit(expftc, pt_bins_centers, scale_vs_pt_calib_train, p0, maxfev=5000)
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
    plt.errorbar(pt_bins_centers, resol_vs_pt_uncalib_valid, xerr=2.5, label=r'Uncalibrated Valid.',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_calib_valid, xerr=2.5,   label=r'Calibrated Valid.',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_uncalib_train, xerr=2.5, label=r'Uncalibrated Train.',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_calib_train, xerr=2.5,   label=r'Calibrated Train.',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
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
        plt.hist(tmpTrain['calib_pt_corr']/tmpTrain['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpTrain['calib_pt_corr']/tmpTrain['gen_pt']), np.std(tmpTrain['calib_pt_corr']/tmpTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpValid['calib_pt_corr']/tmpValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['calib_pt_corr']/tmpValid['gen_pt']), np.std(tmpValid['calib_pt_corr']/tmpValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
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


    # PLOTS LOGLINEARIZER
    # plt.figure(figsize=(10,10))
    # plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfValid['calib_pt_LL']/dfValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated LogLinearized response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt_LL']/dfValid['gen_pt']), np.std(dfValid['calib_pt_LL']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    # plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    # plt.ylabel(r'a.u.')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison_valid_LL.pdf')
    # plt.close()

    # PLOTS LOGLINEARIZER
    # plt.figure(figsize=(10,10))
    # plt.hist(dfTrain['uncalib_pt']/dfTrain['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['uncalib_pt']/dfTrain['gen_pt']), np.std(dfTrain['uncalib_pt']/dfTrain['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfTrain['calib_pt_LL']/dfTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated LogLinearized response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt_LL']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt_LL']/dfTrain['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    # plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    # plt.ylabel(r'a.u.')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison_train_LL.pdf')
    # plt.close()

    # 2D REPOSNSE VS PT
    # plt.figure(figsize=(10,10))
    # plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='red')
    # plt.scatter(dfValid['calib_pt_LL'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated LogLinearized', alpha=0.2, color='green')
    # plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    # plt.ylabel(r'$p_{T}^{Gen \tau}$')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # # plt.xlim(-0.1,5)
    # plt.xlim(0.0,2.0)
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_pt_comparison_LL.pdf')
    # plt.close()

