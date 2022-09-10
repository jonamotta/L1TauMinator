from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.initializers import RandomNormal as RN
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import os

np.random.seed(7)

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

def inspectWeights(model, sparsityPerc):
    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[0].numpy().flatten()
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
    plt.figtext(0.65, 0.82, "{0}% of zeros".format(int(sparsityPerc*100)), wrap=True, horizontalalignment='left',verticalalignment='center', weight='semibold')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/modelSparsity.pdf')
    plt.close()


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                 default=None)
    parser.add_option("--date",         dest="date",                              default=None)
    parser.add_option("--inTag",        dest="inTag",                             default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default="5x9")
    parser.add_option('--sparsity',     dest='sparsity',     type=float,          default=0.5)
    parser.add_option('--train',        dest='train',        action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    sparsityTag = str(options.sparsity).split('.')[0]+'p'+str(options.sparsity).split('.')[1]
    os.system('mkdir -p '+outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 107)
    #       107 are the OHE version of the 35 ieta (absolute) and 72 iphi values
    #
    # Y is (None, 4)
    #       target: visPt, visEta, visPhi, DM

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forCalibrator.npz')['arr_0']


    ############################## Model definition ##############################

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 
    #    - the custom loss targets the visPt of the tau

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/training.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')
        CNN = models.Sequential(name="CNNcalibrator")
        DNN = models.Sequential(name="DNNcalibrator")

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        x = images
        x = layers.Conv2D(16, wndw, input_shape=(N, M, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1")(x)
        x = layers.BatchNormalization(name='BNlayer1')(x)
        x = layers.Activation('relu', name='reluCNNlayer1')(x)
        x = layers.MaxPooling2D(wndw, name="CNNlayer2")(x)
        x = layers.Conv2D(24, wndw, kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3")(x)
        x = layers.BatchNormalization(name='BNlayer2')(x)
        x = layers.Activation('relu', name='reluCNNlayer3')(x)
        x = layers.Flatten(name="CNNflatened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions])
        x = layers.Dense(32, name="DNNlayer1")(x)
        x = layers.Activation('relu', name='reluDNNlayer1')(x)
        x = layers.Dense(16, name="DNNlayer2")(x)
        x = layers.Activation('relu', name='reluDNNlayer2')(x)
        x = layers.Dense(1, name="DNNout")(x)
        TauCalibrated = x

        TauCalibratorModel_base = keras.Model([images, positions], TauCalibrated, name='TauCNNCalibrator')

        # Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs, ending by the 15th epoch
        batch_size = 1024
        NSTEPS = int(X1.shape[0]*0.8)  // batch_size
        def pruneFunction(layer):
            pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                                           final_sparsity = options.sparsity, 
                                                                           begin_step = NSTEPS*2, 
                                                                           end_step = NSTEPS*15, 
                                                                           frequency = NSTEPS)
                             }
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            if isinstance(layer, tf.keras.layers.Dense) and layer.name!='output_dense':
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)  
            return layer

        TauCalibratorModelPruned = tf.keras.models.clone_model(TauCalibratorModel_base, clone_function=pruneFunction)
        callbacks = [pruning_callbacks.UpdatePruningStep()] 

        TauCalibratorModelPruned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.MeanAbsolutePercentageError(),
                                   metrics=['RootMeanSquaredError'],
                                   run_eagerly=True)

        # print(TauCalibrated)
        # print(TauCalibratorModelPruned.summary())
        # exit()
    
        ############################## Model training ##############################

        history = TauCalibratorModelPruned.fit([X1, X2], Y[:,0].reshape(-1,1), epochs=20, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=callbacks)

        TauCalibratorModelPruned.save(outdir + '/TauCNNCalibrator'+sparsityTag+'Pruned')

        for metric in history.history.keys():
            if 'val_' in metric: continue

            plt.plot(history.history[metric], label='Training dataset', lw=2)
            plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            if metric=='loss': plt.legend(loc='upper right')
            else:              plt.legend(loc='lower right')
            mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
            plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/'+metric+'.pdf')
            plt.close()

    else:
        TauCalibratorModelPruned = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag+'/TauCNNCalibrator'+sparsityTag+'Pruned', compile=False)


    ############################## Model validation ##############################

    # load non-pruned model
    TauCalibratorModel = keras.models.load_model(indir+'/TauCNNCalibrator', compile=False)

    X1_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/Y_Calib_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    train_calib = TauCalibratorModel.predict([X1, X2])
    valid_calib = TauCalibratorModel.predict([X1_valid, X2_valid])

    train_calib_pruned = TauCalibratorModelPruned.predict([X1, X2])
    valid_calib_pruned = TauCalibratorModelPruned.predict([X1_valid, X2_valid])
    
    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel()
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['calib_pt_pruned']   = train_calib_pruned.ravel()
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,1].ravel()
    dfTrain['gen_phi']    = Y[:,2].ravel()
    dfTrain['gen_dm']     = Y[:,3].ravel()

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['calib_pt_pruned']   = valid_calib_pruned.ravel()
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,1].ravel()
    dfValid['gen_phi']    = Y_valid[:,2].ravel()
    dfValid['gen_dm']     = Y_valid[:,3].ravel()

    inspectWeights(TauCalibratorModelPruned, options.sparsity)

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt_pruned']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt_pruned']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt_pruned']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt_pruned']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt_pruned']/dfValid['gen_pt']), np.std(dfValid['calib_pt_pruned']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt_pruned']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Pruned Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt_pruned']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt_pruned']/dfTrain['gen_pt'])), color='blue',  ls='--', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt_pruned']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Pruned Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt_pruned']/dfValid['gen_pt']), np.std(dfValid['calib_pt_pruned']/dfValid['gen_pt'])), color='green', ls='--', lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.xlim(0., 2.)
    plt.ylim(0., 3.5)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/responses_comparison_pruning.pdf')
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
    plt.hist(tmp0['uncalib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['uncalib_pt']/tmp0['gen_pt']), np.std(tmp0['uncalib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['uncalib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['uncalib_pt']/tmp1['gen_pt']), np.std(tmp1['uncalib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['uncalib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['uncalib_pt']/tmp10['gen_pt']), np.std(tmp10['uncalib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['uncalib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['uncalib_pt']/tmp11['gen_pt']), np.std(tmp11['uncalib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(tmp0['calib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['calib_pt']/tmp0['gen_pt']), np.std(tmp0['calib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['calib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['calib_pt']/tmp1['gen_pt']), np.std(tmp1['calib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['calib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['calib_pt']/tmp10['gen_pt']), np.std(tmp10['calib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['calib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['calib_pt']/tmp11['gen_pt']), np.std(tmp11['calib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/calibrated_DM_responses_comparison.pdf')
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
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/response_vs_eta_comparison.pdf')
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
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/response_vs_phi_comparison.pdf')
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
    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/response_vs_pt_comparison.pdf')
    plt.close()


    # L1 TO GEN MAPPING
    dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'],
                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                                   # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                                   labels=False,
                                   include_lowest=True)

    dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'],
                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                                   # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                                   labels=False,
                                   include_lowest=True)

    pt_bins_centers = np.arange(17.5,157.5,5)

    trainL1 = dfTrain.groupby('gen_pt_bin')['calib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['calib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['calib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['calib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right')
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['uncalib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right')
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/GenToUncalinL1_pt.pdf')
    plt.close()


    ############################## Feature importance ##############################

#    # since we have two inputs we pass a list of inputs to the explainer
#    explainer = shap.GradientExplainer(TauCalibratorModelPruned, [X1, X2])
#
#    # we explain the model's predictions on the first three samples of the test set
#    shap_values = explainer.shap_values([X1_valid[:3], X2_valid[:3]])
#
#    # since the model has 10 outputs we get a list of 10 explanations (one for each output)
#    print(len(shap_values))
#
#    # since the model has 2 inputs we get a list of 2 explanations (one for each input) for each output
#    print(len(shap_values[0]))
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the first input (this is the feed forward input)
#    shap.image_plot([shap_values[i][0] for i in range(len(shap_values))], X1_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNCalibrator'+sparsityTag+'Pruning_plots/shap1.pdf')
#    plt.close()

# restore normal output
sys.stdout = sys.__stdout__
