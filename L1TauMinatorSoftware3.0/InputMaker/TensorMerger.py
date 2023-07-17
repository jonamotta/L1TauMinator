from optparse import OptionParser
import numpy as np
import zipfile
import pickle
import random
import glob
import math
import sys
import os

# split list l in sublists of length n each
def splitInBlocks (l, n):
    r = len(l) % n

    i = 0
    blocks = []
    while i < len(l):
        if len(blocks)<r:
            blocks.append(l[i:i+n+1])
            i += n+1
        else:
            blocks.append(l[i:i+n])
            i += n

    return blocks


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option("--v",               dest="v",               default=None)
    parser.add_option("--date",            dest="date",            default=None)
    parser.add_option('--caloClNxM',       dest='caloClNxM',       default="5x9")
    parser.add_option("--seedEtCut",       dest="seedEtCut",       default="2p5")
    parser.add_option("--clusteringEtCut", dest="clusteringEtCut", default="")
    parser.add_option("--etaRestriction",  dest="etaRestriction",  default="")
    parser.add_option("--CBCEsplit",       dest="CBCEsplit",       default=1.5, type=float)
    parser.add_option("--uTauPtCut",       dest="uTauPtCut",       default=None,  type=int)
    parser.add_option("--lTauPtCut",       dest="lTauPtCut",       default=None,  type=int)
    parser.add_option("--uEtacut",         dest="uEtacut",         default=None,  type=float)
    parser.add_option("--lEtacut",         dest="lEtacut",         default=None,  type=float)
    parser.add_option('--doBarrel',        dest='doBarrel',        default=False, action='store_true')
    parser.add_option('--doEndcap',        dest='doEndcap',        default=False, action='store_true')
    parser.add_option('--doHH',            dest='doHH',            default=False, action='store_true')
    parser.add_option('--doVBFH',          dest='doVBFH',          default=False, action='store_true')
    parser.add_option('--doGGH',           dest='doGGH',           default=False, action='store_true')
    parser.add_option('--doDY',            dest='doDY',            default=False, action='store_true')
    parser.add_option('--doDYlm',          dest='doDYlm',          default=False, action='store_true')
    parser.add_option('--doMinBias',       dest='doMinBias',       default=False, action='store_true')
    parser.add_option("--filesLim",        dest="filesLim",        default=10000, type=int)
    parser.add_option("--outTag",          dest="outTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    user = os.getcwd().split('/')[5]
    outfile_base = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/'

    inlist_folders = []

    tag = ""
    if options.CBCEsplit : tag += '_CBCEsplit'+str(options.CBCEsplit) 
    if options.uTauPtCut : tag += '_uTauPtCut'+str(options.uTauPtCut) 
    if options.lTauPtCut : tag += '_lTauPtCut'+str(options.lTauPtCut) 
    if options.uEtacut   : tag += '_uEtacut'+str(options.uEtacut) 
    if options.lEtacut   : tag += '_lEtacut'+str(options.lEtacut) 

    if options.doGGH:
        inlist_folders.append(outfile_base+"GluGluHToTauTau_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

    if options.doVBFH:
        inlist_folders.append(outfile_base+"VBFHToTauTau_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

    if options.doDYlm:
        inlist_folders.append(outfile_base+"DYlowmass_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

    if options.doDY:
        inlist_folders.append(outfile_base+"DY_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")


    print(inlist_folders)

    images_files = []
    posits_files = []
    shapes_files = []
    target_files = []

    for i in range(len(inlist_folders)):
        infolder = inlist_folders[i]
        
        if options.doBarrel:
            images_files.extend( glob.glob(infolder+'/barrel/CLTWimages_'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )
            posits_files.extend( glob.glob(infolder+'/barrel/CLTWpositions_'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )
            target_files.extend( glob.glob(infolder+'/barrel/Y_cltw'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )

        if options.doEndcap:
            images_files.extend( glob.glob(infolder+'/endcap/CLTWimages_'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            posits_files.extend( glob.glob(infolder+'/endcap/CLTWpositions_'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            shapes_files.extend( glob.glob(infolder+'/endcap/CL3Dfeatures_cltw'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            target_files.extend( glob.glob(infolder+'/endcap/Y_cltw'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )


    if options.doBarrel:
        images_files.sort()
        posits_files.sort()
        target_files.sort()

        mixer = list(zip(images_files, posits_files, target_files))
        
        for trio in mixer:
            idx1 = trio[0].split('CB_')[1].split('.')[0]
            idx2 = trio[1].split('CB_')[1].split('.')[0]
            idx3 = trio[2].split('CB_')[1].split('.')[0]

            if idx1 != idx2 or idx2 != idx3 or idx1 != idx3:
                print("** ERROR: idexing went wrong")
                print("** ABORTING")

        random.shuffle(mixer)
        images_files, posits_files, target_files = zip(*mixer)
    
    if options.doEndcap:
        images_files.sort()
        posits_files.sort()
        shapes_files.sort()
        target_files.sort()

        mixer = list(zip(images_files, posits_files, shapes_files, target_files))
        
        for quartet in mixer:
            idx1 = quartet[0].split('CE_')[1].split('.')[0]
            idx2 = quartet[1].split('CE_')[1].split('.')[0]
            idx3 = quartet[2].split('CE_')[1].split('.')[0]
            idx4 = quartet[3].split('CE_')[1].split('.')[0]

            if idx1 != idx2 or idx1 != idx3 or idx1 != idx4 or idx2 != idx3 or idx2 != idx4 or idx3 != idx4:
                print("** ERROR: idexing went wrong")
                print("** ABORTING")

        random.shuffle(mixer)
        images_files, posits_files, shapes_files, target_files = zip(*mixer)


    images_toConcat = []
    posits_toConcat = []
    shapes_toConcat = []
    target_toConcat = []

    print('Total number of files:', len(images_files))
    for idx in range(len(images_files)):
        if idx%200==0: print('   ', idx)

        if options.doBarrel:
            images = np.load(images_files[idx], allow_pickle=True)['arr_0']
            posits = np.load(posits_files[idx], allow_pickle=True)['arr_0']
            target = np.load(target_files[idx], allow_pickle=True)['arr_0']

            if target.shape[0] == 0: continue

            # ID mask
            targetID  = target[:,1].reshape(-1,1)
            tau_sel = targetID.reshape(1,-1)[0] > 0
            bkg_sel = targetID.reshape(1,-1)[0] < 1

            # select tau only
            images_tau = images[tau_sel]
            posits_tau = posits[tau_sel]
            target_tau = target[tau_sel]

            # select bkg only
            images_bkg = images[bkg_sel]
            posits_bkg = posits[bkg_sel]
            target_bkg = target[bkg_sel]

            # select random entries from the bkg sample in the same number as the taus
            random_indeces = np.floor(np.random.rand(np.sum(tau_sel))*np.sum(bkg_sel)).astype(int)
            images_bkg_new = images_bkg[random_indeces]
            posits_bkg_new = posits_bkg[random_indeces]
            target_bkg_new = target_bkg[random_indeces]

            # concatenate bkg and tau samples
            images = np.concatenate([images_tau, images_bkg_new])
            posits = np.concatenate([posits_tau, posits_bkg_new])
            target = np.concatenate([target_tau, target_bkg_new])

            images_toConcat.append(images)
            posits_toConcat.append(posits)
            target_toConcat.append(target)

            del targetID, tau_sel, bkg_sel, images, posits, target, images_tau, posits_tau, target_tau, images_bkg, posits_bkg, target_bkg, images_bkg_new, posits_bkg_new, target_bkg_new, random_indeces

        if options.doEndcap:
            images = np.load(images_files[idx], allow_pickle=True)['arr_0']
            posits = np.load(posits_files[idx], allow_pickle=True)['arr_0']
            shapes = np.load(shapes_files[idx], allow_pickle=True)['arr_0']
            target = np.load(target_files[idx], allow_pickle=True)['arr_0']

            if target.shape[0] == 0: continue

            # ID mask
            targetID  = target[:,1].reshape(-1,1)
            tau_sel = targetID.reshape(1,-1)[0] > 0
            bkg_sel = targetID.reshape(1,-1)[0] < 1

            # select tau only
            images_tau = images[tau_sel]
            posits_tau = posits[tau_sel]
            shapes_tau = shapes[tau_sel]
            target_tau = target[tau_sel]

            # select bkg only
            images_bkg = images[bkg_sel]
            posits_bkg = posits[bkg_sel]
            shapes_bkg = shapes[bkg_sel]
            target_bkg = target[bkg_sel]

            # select random entries from the bkg sample in the same number as the taus
            random_indeces = np.floor(np.random.rand(np.sum(tau_sel))*np.sum(bkg_sel)).astype(int)
            images_bkg_new = images_bkg[random_indeces]
            posits_bkg_new = posits_bkg[random_indeces]
            shapes_bkg_new = shapes_bkg[random_indeces]
            target_bkg_new = target_bkg[random_indeces]

            # concatenate bkg and tau samples
            images = np.concatenate([images_tau, images_bkg_new])
            posits = np.concatenate([posits_tau, posits_bkg_new])
            shapes = np.concatenate([shapes_tau, shapes_bkg_new])
            target = np.concatenate([target_tau, target_bkg_new])

            images_toConcat.append(images)
            posits_toConcat.append(posits)
            shapes_toConcat.append(shapes)
            target_toConcat.append(target)

            del targetID, tau_sel, bkg_sel, images, posits, shapes, target, images_tau, posits_tau, shapes_tau, target_tau, images_bkg, posits_bkg, shapes_bkg, target_bkg, images_bkg_new, posits_bkg_new, shapes_bkg_new, target_bkg_new, random_indeces


    dp = int(math.ceil(len(images_toConcat)/5*4))

    if options.doBarrel:
        IMAGES_train = np.concatenate(images_toConcat[:dp])
        POSITS_train = np.concatenate(posits_toConcat[:dp])
        TARGET_train = np.concatenate(target_toConcat[:dp])

        IMAGES_valid = np.concatenate(images_toConcat[dp:])
        POSITS_valid = np.concatenate(posits_toConcat[dp:])
        TARGET_valid = np.concatenate(target_toConcat[dp:])

        # del images_toConcat, posits_toConcat, shapes_toConcat, target_toConcat
        # del images_files, posits_files, shapes_files, target_files 

        print('shape IMAGES_train =', IMAGES_train.shape)
        print('shape POSITS_train =', POSITS_train.shape)
        print('shape TARGET_train =', TARGET_train.shape)
        print('shape IMAGES_valid =', IMAGES_valid.shape)
        print('shape POSITS_valid =', POSITS_valid.shape)
        print('shape TARGET_valid =', TARGET_valid.shape)

    if options.doEndcap:
        IMAGES_train = np.concatenate(images_toConcat[:dp])
        POSITS_train = np.concatenate(posits_toConcat[:dp])
        SHAPES_train = np.concatenate(shapes_toConcat[:dp])
        TARGET_train = np.concatenate(target_toConcat[:dp])

        IMAGES_valid = np.concatenate(images_toConcat[dp:])
        POSITS_valid = np.concatenate(posits_toConcat[dp:])
        SHAPES_valid = np.concatenate(shapes_toConcat[dp:])
        TARGET_valid = np.concatenate(target_toConcat[dp:])

        del images_toConcat, posits_toConcat, shapes_toConcat, target_toConcat
        del images_files, posits_files, shapes_files, target_files 

        print('shape IMAGES_train =', IMAGES_train.shape)
        print('shape POSITS_train =', POSITS_train.shape)
        print('shape SHAPES_train =', SHAPES_train.shape)
        print('shape TARGET_train =', TARGET_train.shape)
        print('shape IMAGES_valid =', IMAGES_valid.shape)
        print('shape POSITS_valid =', POSITS_valid.shape)
        print('shape SHAPES_valid =', SHAPES_valid.shape)
        print('shape TARGET_valid =', TARGET_valid.shape)


    if options.doBarrel:
        os.system('mkdir -p '+outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag)

        np.savez_compressed(outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/images_train.npz', IMAGES_train)
        np.savez_compressed(outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/posits_train.npz', POSITS_train)
        np.savez_compressed(outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/target_train.npz', TARGET_train)

        np.savez_compressed(outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/images_valid.npz', IMAGES_valid)
        np.savez_compressed(outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/posits_valid.npz', POSITS_valid)
        np.savez_compressed(outfile_base+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/target_valid.npz', TARGET_valid)

    if options.doEndcap:
        os.system('mkdir -p '+outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag)

        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/images_train.npz', IMAGES_train)
        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/posits_train.npz', POSITS_train)
        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/shapes_train.npz', SHAPES_train)
        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/target_train.npz', TARGET_train)

        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/images_valid.npz', IMAGES_valid)
        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/posits_valid.npz', POSITS_valid)
        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/shapes_valid.npz', SHAPES_valid)
        np.savez_compressed(outfile_base+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/tensors'+options.outTag+'/target_valid.npz', TARGET_valid)

