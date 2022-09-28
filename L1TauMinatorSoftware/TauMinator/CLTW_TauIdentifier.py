from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
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
    plt.savefig(outdir+'/TauCNNIdentifier_plots/modelSparsity'+which+'.pdf')
    plt.close()


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="5x9")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNIdentifier_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
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
        sys.stdout = Logger(outdir+'/TauCNNIdentifier_plots/training.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        x = images
        x = layers.Conv2D(4, wndw, input_shape=(N, M, 3), kernel_initializer=RN(seed=7), use_bias=False, name="CNNlayer1")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer1')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer1')(x)
        x = layers.MaxPooling2D(wndw, name="MP_CNNlayer1")(x)
        x = layers.Conv2D(8, wndw, kernel_initializer=RN(seed=7), use_bias=False, name="CNNlayer2")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer2')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer2')(x)
        x = layers.Flatten(name="CNNflattened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions])
        x = layers.Dense(16, use_bias=False, name="DNNlayer1")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer1')(x)
        x = layers.Dense(8, use_bias=False, name="DNNlayer2")(x)
        x = layers.Activation('relu', name='RELU_DNNlayer2')(x)
        x = layers.Dense(1, use_bias=False, name="DNNout")(x)
        x = layers.Activation('sigmoid', name='sigmoid_DNNout')(x)
        TauIdentified = x

        TauIdentifierModel = keras.Model([images, positions], TauIdentified, name='TauCNNIdentifier')

        metrics2follow = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(),
                                   metrics=metrics2follow,
                                   run_eagerly=True)

        # print(TauIdentifierModel.summary())
        # exit()

        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauIdentifierModel.fit([X1, X2], Y, epochs=200, batch_size=1024, verbose=1, validation_split=0.25, callbacks=callbacks)

        TauIdentifierModel.save(outdir + '/TauCNNIdentifier')

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauCNNIdentifier_plots/'+metric+'.pdf')
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
                plt.savefig(outdir+'/TauCNNIdentifier_plots/'+metric+'.pdf')
                plt.close()

        ############################## Make split CNN and DNN models ##############################

        image_in = TauIdentifierModel.get_layer(index=0).get_output_at(0)
        flat_out = TauIdentifierModel.get_layer(name='middleMan').get_output_at(0)
        CNNmodel = tf.keras.Model([image_in, positions], flat_out)
        CNNmodel.save(outdir + '/CNNmodel', include_optimizer=False)

        idx = 0
        for layer in TauIdentifierModel.layers:
            if layer._name == 'middleMan': idx += 1; break
            idx += 1
        input_shape = TauIdentifierModel.layers[idx].get_input_shape_at(0)[1]
        CNNflattened = keras.Input(shape=input_shape, name='CNNflattened')
        # create the new nodes for each layer in the path
        x_dnn = CNNflattened
        for layer in TauIdentifierModel.layers[idx:]:
            x_dnn = layer(x_dnn)
        DNNmodel = tf.keras.Model(CNNflattened, x_dnn)
        DNNmodel.save(outdir + '/DNNmodel', include_optimizer=False)

        # validate the full model against the two split models
        y_full  = np.array( TauIdentifierModel.predict([X1, X2]) )
        y_split = np.array( DNNmodel(CNNmodel([X1, X2])) )
        if not np.array_equal(y_full, y_split):
            print('\n\n************************************************************')
            print(" WARNING : Full model and split model outputs do not match")
            print("           Output of np.allclose() = "+str(np.allclose(y_full, y_split)))
            print('************************************************************\n\n')

    else:
        TauIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier', compile=False)

    ############################## Model validation ##############################

    X1_valid = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load(indir+'/Y_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    train_ident = TauIdentifierModel.predict([X1, X2])
    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Y, train_ident)
    AUCtrain = metrics.roc_auc_score(Y, train_ident)

    valid_ident = TauIdentifierModel.predict([X1_valid, X2_valid])
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Y_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Y_valid, valid_ident)

    inspectWeights(TauIdentifierModel, 'kernel')
    # inspectWeights(TauIdentifierModel, 'bias')

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    # plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier_plots/validation_roc.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['true']  = Y_valid.ravel()
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), label='Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), label='PU', color='red', density=True, histtype='step', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier_plots/CNN_score.pdf')
    plt.close()


    ############################## Feature importance ##############################

    # since we have two inputs we pass a list of inputs to the explainer
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
#    plt.savefig(outdir+'/TauCNNIdentifier_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNIdentifier_plots/shap1.pdf')
#    plt.close()

# restore normal output
sys.stdout = sys.__stdout__
