from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks 
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.initializers import RandomNormal as RN
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras import QConv2DBatchnorm, QActivation, QDense
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
from qkeras import qlayers
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cmsml
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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/modelSparsity'+which+'.pdf')
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
    parser.add_option('--sparsity',     dest='sparsity',                         default=0.5, type=float)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training'+options.inTag
    tag = '_QNTZD_sparsity'+str(options.sparsity)
    os.system('mkdir -p '+outdir+'/TauMinator_CB_ident'+tag+'_plots')
    # os.system('mkdir -p '+outdir+'/CMSSWmodels')

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

    # nromalize entries
    # X1 = X1 / 256.
    # X2 = X2 / np.pi

    if options.dm_weighted:
        # create dm weights
        bkg_sel = Yid.reshape(1,-1)[0] < 1
        Ydm = Y[:,4].reshape(-1,1)
        dm0 = Ydm.reshape(1,-1)[0] == 0
        dm1 = (Ydm.reshape(1,-1)[0] == 1) | (Ydm.reshape(1,-1)[0] == 2)
        dm10 = Ydm.reshape(1,-1)[0] == 10
        dm11 = (Ydm.reshape(1,-1)[0] == 11) | (Ydm.reshape(1,-1)[0] == 12)

        bkg_w = Yid.shape[0] / (Yid.shape[0] - (np.sum(dm0)+np.sum(dm1)+np.sum(dm10)+np.sum(dm11)))
        dm0_w = Yid.shape[0] / np.sum(dm0)
        dm1_w = Yid.shape[0] / np.sum(dm1)
        dm10_w = Yid.shape[0] / np.sum(dm10)
        dm11_w = Yid.shape[0] / np.sum(dm11)

        dm_weights = bkg_w*bkg_sel + dm0_w*dm0 + dm1_w*dm1 + dm10_w*dm10 + dm11_w*dm11


    if options.pt_weighted:
        dfweights = pd.DataFrame(np.sum(X1, (1,2,3)).ravel(), columns=['gen_pt'])
        weight_Ebins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 130, 150, 1000]
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
        # plt.xlim(0,175)
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/pt_l1tau.pdf')
        plt.close()


    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauMinator_CB_ident'+tag+'_plots/training_ident.log')
        print(options)

        images = keras.Input(shape = (N, M, 2), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')

        x = QConv2DBatchnorm(4, (2,2), input_shape=(N, M, 2), kernel_initializer=RN(seed=7), use_bias=True,
                                                                     kernel_quantizer='quantized_bits(6,0,alpha=1)', bias_quantizer="quantized_bits(6,0,alpha=1)",
                                                                     name='CNNpBNlayer1')(images)
        x = QActivation('quantized_relu(10,7)', name='RELU_CNNpBNlayer1')(x)
        x = layers.MaxPooling2D((2,2), name='MP_CNNpBNlayer1')(x)
        x = QConv2DBatchnorm(8, (2,2), kernel_initializer=RN(seed=7), use_bias=True,
                                              kernel_quantizer='quantized_bits(6,0,alpha=1)', bias_quantizer="quantized_bits(6,0,alpha=1)",
                                              name='CNNpBNlayer2')(x)
        x = QActivation('quantized_relu(9,6)', name='RELU_CNNpBNlayer2')(x)
        x = layers.Flatten(name="CNNflattened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions])
        
        y = QDense(16, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name='IDlayer1')(x)
        y = QActivation('quantized_relu(9,1)', name='RELU_IDlayer1')(y)
        y = QDense(8, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name='IDlayer2')(y)
        y = QActivation('quantized_relu(8,1)', name='RELU_IDlayer2')(y)
        y = QDense(1, use_bias=False, kernel_quantizer='quantized_bits(6,0,alpha=1)', name="IDout")(y)
        y = layers.Activation('sigmoid', name='sigmoid_IDout')(y)

        TauIdentified = y

        TauMinatorModel = keras.Model(inputs=[images, positions], outputs=TauIdentified, name='TauMinator_CB_indent')

        # Prune all convolutional and dense layers gradually from 0 to X% sparsity every 2 epochs, ending by the 15th epoch
        batch_size = 1024
        NSTEPS = int(X1.shape[0]*0.75) // batch_size
        def pruneFunction(layer):
            pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                                           final_sparsity = options.sparsity, 
                                                                           begin_step = NSTEPS*2, 
                                                                           end_step = NSTEPS*25, 
                                                                           frequency = NSTEPS)
                             }
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            if isinstance(layer, tf.keras.layers.Dense) and layer.name!='DNNout':
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)  
            return layer

        TauMinatorModelPruned = tf.keras.models.clone_model(TauMinatorModel, clone_function=pruneFunction)

        if options.dm_weighted or options.pt_weighted:
            TauMinatorModelPruned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                                    sample_weight_mode='sample-wise',
                                    run_eagerly=True)
        else:
            TauMinatorModelPruned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                                    run_eagerly=True)

        # print(TauMinatorModelPruned.summary())
        # exit()

        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
                     pruning_callbacks.UpdatePruningStep()]

        if options.dm_weighted:
            history = TauMinatorModelPruned.fit([X1, X2], Yid, epochs=500, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks, sample_weight=dm_weights)

        elif options.pt_weighted:
            history = TauMinatorModelPruned.fit([X1, X2], Yid, epochs=500, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks, sample_weight=pt_weights)
        
        else:
            history = TauMinatorModelPruned.fit([X1, X2], Yid, epochs=500, batch_size=1024, shuffle=True, verbose=1, validation_split=0.10, callbacks=callbacks)

        TauMinatorModelPruned = strip_pruning(TauMinatorModelPruned)

        TauMinatorModelPruned.save(outdir + '/TauMinator_CB_ident'+tag)

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/'+metric+'.pdf')
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
                plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/'+metric+'.pdf')
                plt.close()

        ############################## Make split CNN and DNN models ##############################

        # save convolutional part alone
        image_in = TauMinatorModelPruned.get_layer(index=0).get_output_at(0)
        posit_in = TauMinatorModelPruned.get_layer(name='TowerClusterPosition').get_output_at(0)
        
        ## build CNN only model without concatenation to use for vivado accelerator
        x_cnn = image_in
        for layer in TauMinatorModelPruned.layers:
            if layer.__class__.__name__ == "InputLayer": continue
            if layer.__class__.__name__ == "Concatenate": break
            x_cnn = layer(x_cnn)
        CNNmodel = tf.keras.Model([image_in], x_cnn, name="TauMinator_CB_conv")
        CNNmodel.save(outdir+'/CNNmodel'+tag+'_special4accelerator', include_optimizer=False)

        ## build CNN only model with concatenation to use for other purposes
        x_cnn = image_in
        idx = 0
        for layer in TauMinatorModelPruned.layers:
            if layer.__class__.__name__ == "InputLayer": continue
            if layer.__class__.__name__ == "Concatenate":
                x_cnn = layer([x_cnn,posit_in])
                break
            x_cnn = layer(x_cnn)
        CNNmodel = tf.keras.Model([image_in, posit_in], x_cnn, name="TauMinator_CB_conv")
        CNNmodel.save(outdir+'/CNNmodel'+tag, include_optimizer=False)
        # cmsml.tensorflow.save_graph(outdir+'/CMSSWmodels/CNNmodel_CB.pb', CNNmodel, variables_to_constants=True)
        # cmsml.tensorflow.save_graph(outdir+'/CMSSWmodels/CNNmodel_CB.pb.txt', CNNmodel, variables_to_constants=True)

        ## create middleman to be used for the saving of the id dnn alone
        idx = 0
        for layer in TauMinatorModelPruned.layers:
            if layer._name == 'middleMan': idx += 1; break
            idx += 1
        input_shape = TauMinatorModelPruned.layers[idx].get_input_shape_at(0)[1]
        middleMan = keras.Input(shape=input_shape, name='middleMan')
        
        # save id dense part alone
        x_dnn = middleMan
        for layer in TauMinatorModelPruned.layers[idx:]:
            x_dnn = layer(x_dnn)
        ID_DNNmodel = tf.keras.Model(middleMan, x_dnn, name='TauMinator_CB_ident')
        ID_DNNmodel.save(outdir+'/ID_DNNmodel'+tag, include_optimizer=False)
        # cmsml.tensorflow.save_graph(outdir+'/CMSSWmodels/DNNident_CB.pb', ID_DNNmodel, variables_to_constants=True)
        # cmsml.tensorflow.save_graph(outdir+'/CMSSWmodels/DNNident_CB.pb.txt', ID_DNNmodel, variables_to_constants=True)

        # validate the full model against the two split models
        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)

        # nromalize entries
        # X1_valid = X1_valid / 256.
        # X2_valid = X2_valid / np.pi

        y_full = np.array( TauMinatorModelPruned.predict([X1_valid, X2_valid]) )
        y_split  = np.array( ID_DNNmodel(CNNmodel([X1_valid, X2_valid])) )
        if not np.array_equal(y_full, y_split):
            print('\n\n****************************************************************')
            print(" WARNING : Full model and split ID model outputs do not match")
            print("           Output of np.allclose() = "+str(np.allclose(y_full, y_split)))
            print('****************************************************************\n\n')

        # restore normal output
        sys.stdout = sys.__stdout__

    else:
        TauMinatorModelPruned = keras.models.load_model(outdir+'/TauMinator_CB_ident'+tag, compile=False)

        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)

        # nromalize entries
        # X1_valid = X1_valid / 256.
        # X2_valid = X2_valid / np.pi

    ############################## Model validation and pots ##############################

    train_ident = TauMinatorModelPruned.predict([X1, X2])
    valid_ident = TauMinatorModelPruned.predict([X1_valid, X2_valid])

    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Yid, train_ident)
    AUCtrain = metrics.roc_auc_score(Yid, train_ident)

    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Yid_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Yid_valid, valid_ident)

    # save ID working points
    WP99 = np.interp(0.99, TPRvalid, THRvalid)
    WP95 = np.interp(0.95, TPRvalid, THRvalid)
    WP90 = np.interp(0.90, TPRvalid, THRvalid)
    WP85 = np.interp(0.85, TPRvalid, THRvalid)
    WP80 = np.interp(0.80, TPRvalid, THRvalid)
    WP75 = np.interp(0.75, TPRvalid, THRvalid)
    wp_dict = {
        'wp99' : WP99,
        'wp95' : WP95,
        'wp90' : WP90,
        'wp85' : WP85,
        'wp80' : WP80,
        'wp75' : WP75
    }
    save_obj(wp_dict, outdir+'/TauMinator_CB_ident'+tag+'_plots/CLTW_TauIdentifier_WPs.pkl')

    inspectWeights(TauMinatorModelPruned, 'kernel')
    # inspectWeights(TauMinatorModelPruned, 'bias')

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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.yscale('log')
    plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_log.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['true']  = Yid_valid.ravel()
    df['gen_pt'] = Y_valid[:,0].ravel()
    df['L1_et'] = np.sum(X1_valid, (1,2,3)).ravel() #* 256.
    # df['gen_dm'] = Y_valid[:,4].ravel()
    
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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/CNN_score.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(df[df['true']==0]['score'], df[df['true']==0]['L1_et'], color='red', alpha=0.2)
    plt.scatter(df[df['true']==1]['score'], df[df['true']==1]['L1_et'], color='green', alpha=0.2)
    plt.grid(linestyle=':')
    # plt.xlim(0.5,1.001)
    plt.ylim(17,40)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel(r'CNN score')
    plt.ylabel(r'$p_{T}^{Gen. \tau}$')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/CNN_score_vs_l1et.pdf')
    plt.close()

    ############################## Compare to non-pruned non-quantized model ##############################

    fullCNN = keras.models.load_model(outdir.replace('QNTZD','')+'/TauMinator_CB_ident_sparsity0.0', compile=False)

    full_valid_ident = fullCNN.predict([X1_valid, X2_valid])
    full_FPRvalid, full_TPRvalid, full_THRvalid = metrics.roc_curve(Yid_valid, full_valid_ident)
    full_AUCvalid = metrics.roc_auc_score(Yid_valid, full_valid_ident)

    plt.figure(figsize=(10,10))
    plt.plot(TPRvalid, FPRvalid, label='Quantized model ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.plot(full_TPRvalid, full_FPRvalid, label='Full model ROC, AUC = %.3f' % (full_AUCvalid), color='red',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    # plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_vsFull.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRvalid, FPRvalid, label='Quantized model ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.plot(full_TPRvalid, full_FPRvalid, label='Full model ROC, AUC = %.3f' % (full_AUCvalid), color='red',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.yscale('log')
    plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_vsFull_log.pdf')
    plt.close()

    fulldf = pd.DataFrame()
    fulldf['score'] = full_valid_ident.ravel()
    fulldf['true']  = Yid_valid.ravel()
    fulldf['gen_pt'] = Y_valid[:,0].ravel()
    fulldf['L1_et'] = np.sum(X1_valid, (1,2,3)).ravel() #* 256.
    # df['gen_dm'] = Y_valid[:,4].ravel()

    plt.figure(figsize=(10,10))
    plt.hist(fulldf[fulldf['true']==1]['score'], bins=np.arange(0,1,0.05), label='Full model - Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(fulldf[fulldf['true']==0]['score'], bins=np.arange(0,1,0.05), label='Full model - PU', color='red', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), ls='--', label='Quantized model - Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), ls='--', label='Quantized model - PU', color='red', density=True, histtype='step', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/CNN_score_vsFull.pdf')
    plt.close()

    ############################## Model validation at high pt ##############################

    sel = np.sum(X1, (1,2,3)).ravel() > 50#/256.
    X1 = X1[sel]
    X2 = X2[sel]
    Yid = Yid[sel]

    sel_valid = np.sum(X1_valid, (1,2,3)).ravel() > 50#/256.
    X1_valid = X1_valid[sel_valid]
    X2_valid = X2_valid[sel_valid]
    Yid_valid = Yid_valid[sel_valid]

    train_ident = TauMinatorModelPruned.predict([X1, X2])
    valid_ident = TauMinatorModelPruned.predict([X1_valid, X2_valid])

    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Yid, train_ident)
    AUCtrain = metrics.roc_auc_score(Yid, train_ident)

    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Yid_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Yid_valid, valid_ident)

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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_geq50.pdf')
    plt.close()

    sel = np.sum(X1, (1,2,3)).ravel() > 75#/256.
    X1 = X1[sel]
    X2 = X2[sel]
    Yid = Yid[sel]

    sel_valid = np.sum(X1_valid, (1,2,3)).ravel() > 75#/256.
    X1_valid = X1_valid[sel_valid]
    X2_valid = X2_valid[sel_valid]
    Yid_valid = Yid_valid[sel_valid]

    train_ident = TauMinatorModelPruned.predict([X1, X2])
    valid_ident = TauMinatorModelPruned.predict([X1_valid, X2_valid])

    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Yid, train_ident)
    AUCtrain = metrics.roc_auc_score(Yid, train_ident)

    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Yid_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Yid_valid, valid_ident)

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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_geq75.pdf')
    plt.close()

    sel = np.sum(X1, (1,2,3)).ravel() > 100#/256.
    X1 = X1[sel]
    X2 = X2[sel]
    Yid = Yid[sel]

    sel_valid = np.sum(X1_valid, (1,2,3)).ravel() > 100#/256.
    X1_valid = X1_valid[sel_valid]
    X2_valid = X2_valid[sel_valid]
    Yid_valid = Yid_valid[sel_valid]

    train_ident = TauMinatorModelPruned.predict([X1, X2])
    valid_ident = TauMinatorModelPruned.predict([X1_valid, X2_valid])

    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Yid, train_ident)
    AUCtrain = metrics.roc_auc_score(Yid, train_ident)

    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Yid_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Yid_valid, valid_ident)

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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_geq100.pdf')
    plt.close()

    sel = np.sum(X1, (1,2,3)).ravel() > 125#/256.
    X1 = X1[sel]
    X2 = X2[sel]
    Yid = Yid[sel]

    sel_valid = np.sum(X1_valid, (1,2,3)).ravel() > 125#/256.
    X1_valid = X1_valid[sel_valid]
    X2_valid = X2_valid[sel_valid]
    Yid_valid = Yid_valid[sel_valid]

    train_ident = TauMinatorModelPruned.predict([X1, X2])
    valid_ident = TauMinatorModelPruned.predict([X1_valid, X2_valid])

    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Yid, train_ident)
    AUCtrain = metrics.roc_auc_score(Yid, train_ident)

    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Yid_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Yid_valid, valid_ident)

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
    plt.savefig(outdir+'/TauMinator_CB_ident'+tag+'_plots/validation_roc_geq125.pdf')
    plt.close()

























