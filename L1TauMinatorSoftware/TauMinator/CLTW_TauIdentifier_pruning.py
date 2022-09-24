from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.initializers import RandomNormal as RN
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

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

def inspectWeights(model, sparsityPerc, which):
    if which=='kernel': idx=0
    if which=='bias':   idx=1

    perc = 0 ; cnt = 0
    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[idx].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))
        if np.sum(weights==0)/np.size(weights) != 0: perc += np.sum(weights==0)/np.size(weights) ; cnt += 1

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
    plt.figtext(0.65, 0.82, "~{0}% of zeros".format(int(perc/cnt*100)), wrap=True, horizontalalignment='left',verticalalignment='center', fontsize=18)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/modelSparsity'+which+'.pdf')
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

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    sparsityTag = str(options.sparsity).split('.')[0]+'p'+str(options.sparsity).split('.')[1]
    os.system('mkdir -p '+outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 107)
    #       107 are the OHE version of the 35 ieta (absolute) and 72 iphi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    Y = np.load(indir+'/Y_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']


    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/training.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')
        CNN = models.Sequential(name="CNNidentifier")
        DNN = models.Sequential(name="DNNidentifier")

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        x = images
        x = layers.Conv2D(16, wndw, input_shape=(N, M, 3), kernel_initializer=RN(seed=7), use_bias=False, name="CNNlayer1")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer1')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer1')(x)
        x = layers.MaxPooling2D(wndw, name="MP_CNNlayer1")(x)
        x = layers.Conv2D(24, wndw, kernel_initializer=RN(seed=7), use_bias=False, name="CNNlayer2")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer2')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer2')(x)
        x = layers.Flatten(name="CNNflatened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions])
        x = layers.Dense(32, use_bias=False, name="DNNlayer1")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer1')(x)
        x = layers.Dense(16, use_bias=False, name="DNNlayer2")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer2')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
        TauIdentified = x

        TauIdentifierModel_base = keras.Model([images, positions], TauIdentified, name='TauCNNIdentifier')

        # Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs, ending by the 15th epoch
        batch_size = 1024
        NSTEPS = int(X1.shape[0]*0.75)  // batch_size
        def pruneFunction(layer):
            pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                                           final_sparsity = options.sparsity, 
                                                                           begin_step = NSTEPS*2, 
                                                                           end_step = NSTEPS*30, 
                                                                           frequency = NSTEPS)
                             }
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            if isinstance(layer, tf.keras.layers.Dense) and layer.name!='DNNout':
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)  
            return layer

        TauIdentifierModelPruned = tf.keras.models.clone_model(TauIdentifierModel_base, clone_function=pruneFunction)

        metrics2follow = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        TauIdentifierModelPruned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                         loss=tf.keras.losses.BinaryCrossentropy(),
                                         metrics=metrics2follow,
                                         run_eagerly=True)


        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
                     pruning_callbacks.UpdatePruningStep()]
        
        history = TauIdentifierModelPruned.fit([X1, X2], Y, epochs=200, batch_size=batch_size, verbose=1, validation_split=0.25, callbacks=callbacks)
        
        TauIdentifierModelPruned = strip_pruning(TauIdentifierModelPruned)
        
        TauIdentifierModelPruned.save(outdir + '/TauCNNIdentifier'+sparsityTag+'Pruned')

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/'+metric+'.pdf')
                plt.close()

            else:
                if 'val_' in metric: continue

                plt.plot(history.history[metric], label='Training dataset', lw=2)
                plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
                plt.xlabel('Epoch')
                if metric=='loss':
                    plt.ylabel('Loss')
                    plt.legend(loc='upper right')
                elif metric=='auc':
                    plt.ylabel('AUC')
                    plt.legend(loc='lower right')
                elif metric=='binary_accuracy':
                    plt.ylabel('Binary accuracy')
                    plt.legend(loc='lower right')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/'+metric+'.pdf')
                plt.close()

        ############################## Make split CNN and DNN models ##############################

        # create the new nodes for each layer in the path
        x_cnn = images
        for layer in TauIdentifierModelPruned.layers[1:9]:
            x_cnn = layer(x_cnn)
        x_cnn = TauIdentifierModelPruned.layers[10]([x_cnn,positions])
        CNNmodel = tf.keras.Model([images, positions], x_cnn)
        CNNmodel.save(outdir + '/CNNmodel'+sparsityTag+'Pruned', include_optimizer=False)

        idx = 0
        for layer in TauIdentifierModelPruned.layers:
            if layer._name == 'middleMan': idx += 1; break
            idx += 1
        input_shape = TauIdentifierModelPruned.layers[idx].get_input_shape_at(0)[1]
        CNNflattened = keras.Input(shape=input_shape, name='CNNflattened')
        # create the new nodes for each layer in the path
        x_dnn = CNNflattened
        for layer in TauIdentifierModelPruned.layers[idx:]:
            x_dnn = layer(x_dnn)
        DNNmodel = tf.keras.Model(CNNflattened, x_dnn)
        DNNmodel.save(outdir + '/DNNmodel'+sparsityTag+'Pruned', include_optimizer=False)

        # validate the full model against the two split models
        y_full  = np.array( TauIdentifierModelPruned.predict([X1, X2]) )
        y_split = np.array( DNNmodel(CNNmodel([X1, X2])) )
        if not np.array_equal(y_full, y_split):
            print('\n\n************************************************************')
            print(" WARNING : Full model and split model outputs do not match")
            print("           Output of np.allclose() = "+str(np.allclose(y_full, y_split)))
            print('************************************************************\n\n')

    else:
        TauIdentifierModelPruned = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier'+sparsityTag+'Pruned', compile=False)

    ############################## Model validation ##############################

    # load non-pruned model
    TauIdentifierModel = keras.models.load_model(indir+'/TauCNNIdentifier', compile=False)

    X1_valid = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load(indir+'/Y_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    train_ident = TauIdentifierModel.predict([X1, X2])
    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Y, train_ident)
    AUCtrain = metrics.roc_auc_score(Y, train_ident)

    valid_ident = TauIdentifierModel.predict([X1_valid, X2_valid])
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Y_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Y_valid, valid_ident)

    train_ident_pruned = TauIdentifierModelPruned.predict([X1, X2])
    FPRtrainPruned, TPRtrainPruned, THRtrainPruned = metrics.roc_curve(Y, train_ident_pruned)
    AUCtrain_pruned = metrics.roc_auc_score(Y, train_ident_pruned)

    valid_ident_pruned = TauIdentifierModelPruned.predict([X1_valid, X2_valid])
    FPRvalidPruned, TPRvalidPruned, THRvalidPruned = metrics.roc_curve(Y_valid, valid_ident_pruned)
    AUCvalid_pruned = metrics.roc_auc_score(Y_valid, valid_ident_pruned)

    inspectWeights(TauIdentifierModelPruned, options.sparsity, 'kernel')
    # inspectWeights(TauIdentifierModelPruned, options.sparsity, 'bias')

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain), color='blue', lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green', lw=2)
    plt.plot(TPRtrainPruned, FPRtrainPruned, label='Training ROC - Pruned, AUC = %.3f' % (AUCtrain_pruned), color='blue', lw=2, ls='--')
    plt.plot(TPRvalidPruned, FPRvalidPruned, label='Validation ROC - Pruned, AUC = %.3f' % (AUCvalid_pruned), color='green', lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/validation_roc.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain), color='blue', lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green', lw=2)
    plt.plot(TPRtrainPruned, FPRtrainPruned, label='Training ROC - Pruned, AUC = %.3f' % (AUCtrain_pruned), color='blue', lw=2, ls='--')
    plt.plot(TPRvalidPruned, FPRvalidPruned, label='Validation ROC - Pruned, AUC = %.3f' % (AUCvalid_pruned), color='green', lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.ylim(0.03,1)
    plt.yscale('log')
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/validation_roc_zoomed.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['scorePruned'] = valid_ident_pruned.ravel()
    df['true']  = Y_valid.ravel()
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), label='Tau', color='green', density=True, lw=2, histtype='step')
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), label='PU', color='red', density=True, lw=2, histtype='step')
    plt.hist(df[df['true']==1]['scorePruned'], bins=np.arange(0,1,0.05), label='Tau - Pruned Model', color='green', density=True, lw=2, ls='--', histtype='step')
    plt.hist(df[df['true']==0]['scorePruned'], bins=np.arange(0,1,0.05), label='PU - Pruned Model', color='red', density=True, lw=2, ls='--', histtype='step')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/CNN_score.pdf')
    plt.close()


    ############################## Feature importance ##############################

#    # since we have two inputs we pass a list of inputs to the explainer
#    explainer = shap.GradientExplainer(TauIdentifierModel, [X1, X2])
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
#    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
#    plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
#    plt.savefig(outdir+'/TauCNNIdentifier'+sparsityTag+'Pruning_plots/shap1.pdf')
#    plt.close()

# restore normal output
sys.stdout = sys.__stdout__
