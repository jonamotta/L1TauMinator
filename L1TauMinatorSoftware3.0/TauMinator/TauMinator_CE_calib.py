# from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
# from tensorflow_model_optimization.sparsity.keras import strip_pruning
# from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.initializers import RandomNormal as RN
import tensorflow_model_optimization as tfmot
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
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/modelSparsity'+which+'.pdf')
    plt.close()

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)


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
    parser.add_option('--sparsity',     dest='sparsity',                         default=0.5, type=float)
    parser.add_option('--sparsityCNN',  dest='sparsityCNN',                      default=0.0, type=float)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training'+options.inTag
    tag = '_sparsity'+str(options.sparsity)
    os.system('mkdir -p '+outdir+'/TauMinator_CE_calib'+tag+'_plots')
    os.system('mkdir -p '+outdir+'/CMSSWmodels'+tag)

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    # 
    # X3 is (None, 17)
    #       12 are the shape variables of the cl3ds
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(outdir+'/tensors/images_train.npz')['arr_0']
    X2 = np.load(outdir+'/tensors/posits_train.npz')['arr_0']
    X3 = np.load(outdir+'/tensors/shapes_train.npz')['arr_0']
    Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
    Yid  = Y[:,1].reshape(-1,1)
    Ycal = Y[:,0].reshape(-1,1)

    X1[X1 < 1.] = 0.0

    # select only taus
    tau_sel = Yid.reshape(1,-1)[0] > 0
    X1 = X1[tau_sel]
    X2 = X2[tau_sel]
    X3 = X3[tau_sel]
    Y = Y[tau_sel]
    Ycal = Ycal[tau_sel]

    # select only below 150 GeV
    pt_sel = Ycal.reshape(1,-1)[0] < 150
    X1 = X1[pt_sel]
    X2 = X2[pt_sel]
    X3 = X3[pt_sel]
    Y = Y[pt_sel]
    Ycal = Ycal[pt_sel]

    # select not too weird reconstruuctions
    X1_totE = np.sum(X1, (1,2,3)).reshape(-1,1)
    Yfactor = Ycal / X1_totE
    farctor_sel = (Yfactor.reshape(1,-1)[0] < 2.4) & (Yfactor.reshape(1,-1)[0] > 0.4)
    X1 = X1[farctor_sel]
    X2 = X2[farctor_sel]
    X3 = X3[farctor_sel]
    Y = Y[farctor_sel]
    Ycal = Ycal[farctor_sel]
    Yfactor = Yfactor[farctor_sel]

    cl3d_pt_uncalib_train = X3[:,0]

    # nromalize entries
    # X1 = X1 / 256.
    # X2 = X2 / np.pi
    # Ycal = Ycal / 256.
    scaler = load_obj(indir+'/CL3D_features_scaler/cl3d_features_scaler.pkl')
    X3 = scaler.transform(X3)

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

    if options.pt_weighted:
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
        # dfweights['weight'] = dfweights.apply(lambda row: customPtDownWeight(row) , axis=1)
        dfweights['weight'] = dfweights['weight'] / np.mean(dfweights['weight'])
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
        plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/pt_gentau.pdf')
        plt.close()

        del dfweights

    # features2use = pt, eta, showerlength, coreshowerlength, spptot, szz, srrtot, meanz
    features2use = [0, 2, 4, 5, 9, 11, 12, 16]
    X3 = X3[:,features2use]

    ############################## Models definition ##############################

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position and the cl3ds features

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauMinator_CE_calib'+tag+'_plots/training_calib.log')
        print(options)

        middleMan = keras.Input(shape=24+2+len(features2use), name='middleMan')

        x = layers.Dense(64, use_bias=False, name="DNNlayer1")(middleMan)
        x = layers.Activation('relu', name='RELU_DNNlayer1')(x)
        x = layers.Dense(64, use_bias=False, name="DNNlayer2")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer2')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        x = layers.Activation('relu', name='LIN_DNNout')(x)

        # x = layers.Dense(16, use_bias=False, name="DNNlayer1")(middleMan)
        # x = layers.Activation('relu', name='RELU_DNNlayer1')(x)
        # x = layers.Dense(8, use_bias=False, name="DNNlayer2")(x)
        # x = layers.Activation('relu', name='RELU_DNNlayer2')(x)
        # x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        
        TauCalibrated = x

        TauMinatorModelPruned = keras.Model(inputs=middleMan, outputs=TauCalibrated, name='TauMinator_CE_calib')

        # Prune all convolutional and dense layers gradually from 0 to X% sparsity every 2 epochs, ending by the 15th epoch
        # batch_size = 1024
        # NSTEPS = int(X1.shape[0]*0.75) // batch_size
        # def pruneFunction(layer):
        #     pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
        #                                                                    final_sparsity = options.sparsity, 
        #                                                                    begin_step = NSTEPS*2, 
        #                                                                    end_step = NSTEPS*25, 
        #                                                                    frequency = NSTEPS)
        #                      }
        #     if isinstance(layer, tf.keras.layers.Conv2D):
        #         return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        #     if isinstance(layer, tf.keras.layers.Dense) and layer.name!='DNNout':
        #         return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)  
        #     return layer

        # TauMinatorModelPruned = tf.keras.models.clone_model(TauMinatorModel, clone_function=pruneFunction)

        if options.dm_weighted or options.pt_weighted:
            TauMinatorModelPruned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                    metrics=['RootMeanSquaredError'],
                                    sample_weight_mode='sample-wise',
                                    run_eagerly=True)

        else:
            TauMinatorModelPruned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                    metrics=['RootMeanSquaredError'],
                                    run_eagerly=True)

        # print(TauMinatorModelPruned.summary())
        # exit()

        ############################## Model training ##############################

        CNN = keras.models.load_model(outdir+'/CNNmodel_sparsity'+str(options.sparsityCNN), compile=False)
        CNNprediction = CNN([X1, X2]) #, X3])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]
                     # pruning_callbacks.UpdatePruningStep()]

        if options.dm_weighted:
            history = TauMinatorModelPruned.fit(CNNprediction, Ycal, epochs=500, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks, sample_weight=dm_weights)

        elif options.pt_weighted:
            history = TauMinatorModelPruned.fit(CNNprediction, Ycal, epochs=500, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks, sample_weight=pt_weights)

        else:
            history = TauMinatorModelPruned.fit(CNNprediction, Ycal, epochs=500, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks)

        # TauMinatorModelPruned = strip_pruning(TauMinatorModelPruned)

        TauMinatorModelPruned.save(outdir+'/CAL_DNNmodel'+tag, include_optimizer=False)
        cmsml.tensorflow.save_graph(outdir+'/CMSSWmodels'+tag+'/DNNcalib_CE.pb', TauMinatorModelPruned, variables_to_constants=True)
        cmsml.tensorflow.save_graph(outdir+'/CMSSWmodels'+tag+'/DNNcalib_CE.pb.txt', TauMinatorModelPruned, variables_to_constants=True)

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/'+metric+'.pdf')
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
                plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/'+metric+'.pdf')
                plt.close()

        # restore normal output
        sys.stdout = sys.__stdout__

    else:
        CNN = keras.models.load_model(outdir+'/CNNmodel', compile=False)
        TauMinatorModelPruned = keras.models.load_model(outdir+'/CAL_DNNmodel'+tag, compile=False)

    ############################## Model validation and pots ##############################

    X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
    X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
    X3_valid = np.load(outdir+'/tensors/shapes_valid.npz')['arr_0']
    Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
    Yid_valid  = Y_valid[:,1].reshape(-1,1)
    Ycal_valid = Y_valid[:,0].reshape(-1,1)

    # select only taus
    tau_sel_valid = Yid_valid.reshape(1,-1)[0] > 0
    X1_valid = X1_valid[tau_sel_valid]
    X2_valid = X2_valid[tau_sel_valid]
    X3_valid = X3_valid[tau_sel_valid]
    Y_valid = Y_valid[tau_sel_valid]
    Ycal_valid = Ycal_valid[tau_sel_valid]

    # select only below 150 GeV
    pt_sel = Ycal_valid.reshape(1,-1)[0] < 150
    X1_valid = X1_valid[pt_sel]
    X2_valid = X2_valid[pt_sel]
    X3_valid = X3_valid[pt_sel]
    Y_valid = Y_valid[pt_sel]
    Ycal_valid = Ycal_valid[pt_sel]

    cl3d_pt_uncalib_valid = X3_valid[:,0]

    # nromalize entries
    # X1_valid = X1_valid / 256.
    # X2_valid = X2_valid / np.pi
    # Ycal_valid = Ycal_valid / 256.
    X3_valid = scaler.transform(X3_valid)
    X3_valid = X3_valid[:,features2use]

    train_calib = TauMinatorModelPruned.predict(CNN([X1, X2, X3])) #* 256.
    valid_calib = TauMinatorModelPruned.predict(CNN([X1_valid, X2_valid, X3_valid])) #* 256.

    inspectWeights(TauMinatorModelPruned, 'kernel')

    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel() #* 256.
    dfTrain['cl3d_pt']    = cl3d_pt_uncalib_train.ravel()
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,2].ravel()
    dfTrain['gen_phi']    = Y[:,3].ravel()
    # dfTrain['gen_dm']     = Y[:,4].ravel()

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel() #* 256.
    dfValid['cl3d_pt']    = cl3d_pt_uncalib_valid.ravel()
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,2].ravel()
    dfValid['gen_phi']    = Y_valid[:,3].ravel()
    # dfValid['gen_dm']     = Y_valid[:,4].ravel()

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'$CL^{5\times9}$ uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['cl3d_pt']/dfValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'$CL^{3D}$ uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['cl3d_pt']/dfValid['gen_pt']), np.std(dfValid['cl3d_pt']/dfValid['gen_pt'])),  color='red',  ls ='--', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/responses_comparison.pdf')
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
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/response_vs_eta_comparison.pdf')
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
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/response_vs_phi_comparison.pdf')
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
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/response_vs_pt_comparison.pdf')
    plt.close()


    # L1 TO GEN MAPPING
    bins = np.arange(18,151,3)
    # bins = np.append(bins, [2000])
    pt_bins_centers = np.array(np.arange(19.5,150,3))
    # pt_bins_centers = np.append(pt_bins_centers, [350])

    dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'], bins=bins, labels=False, include_lowest=True)
    dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'], bins=bins, labels=False, include_lowest=True)

    trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].mean())
    validL1 = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].mean())
    trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].std())
    validL1std = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].std())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.plot(np.array(np.arange(0,200,1)), np.array(np.arange(0,200,1)), label='Ideal calibration', color='black', ls='--', lw=2)
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(17, 151)
    plt.ylim(17, 151)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean())
    validL1 = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean())
    trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std())
    validL1std = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].std())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.plot(np.array(np.arange(0,200,1)), np.array(np.arange(0,200,1)), label='Ideal calibration', color='black', ls='--', lw=2)
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(17, 151)
    plt.ylim(17, 151)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/GenToUncalinL1_pt.pdf')
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
        plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/responses_comparison_'+str(ledge)+'pt'+str(uedge)+'.pdf')
        plt.close()

        scale_vs_pt_uncalib.append(np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_uncalib.append(np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

        scale_vs_pt_calib.append(np.mean(tmpValid['calib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_calib.append(np.std(tmpValid['calib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))


    # scale vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, scale_vs_pt_uncalib, xerr=2.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_calib, xerr=2.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'Scale')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(17, 151)
    plt.ylim(0.5,1.5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/scale_vs_pt.pdf')
    plt.close()

    # resolution vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, resol_vs_pt_uncalib, xerr=2.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_calib, xerr=2.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'Resolution')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(17, 151)
    plt.ylim(0.1,0.4)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_calib'+tag+'_plots/resolution_vs_pt.pdf')
    plt.close()


    '''
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
    plt.savefig(outdir+'/TauMinator_CE_calib_plots/uncalibrated_DM_responses_comparison.pdf')
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
    plt.savefig(outdir+'/TauMinator_CE_calib_plots/calibrated_DM_responses_comparison.pdf')
    plt.close()
'''

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
    # plt.savefig(outdir+'/TauMinator_CE_calib_plots/responses_comparison_valid_LL.pdf')
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
    # plt.savefig(outdir+'/TauMinator_CE_calib_plots/responses_comparison_train_LL.pdf')
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
    # plt.savefig(outdir+'/TauMinator_CE_calib_plots/response_vs_pt_comparison_LL.pdf')
    # plt.close()

