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
    parser.add_option("--v",              dest="v",              default=None)
    parser.add_option("--date",           dest="date",           default=None)
    parser.add_option('--caloClNxM',      dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",      dest="seedEtCut",      default="2p5")
    parser.add_option('--doBarrel',       dest='doBarrel',       default=False, action='store_true')
    parser.add_option('--doEndcap',       dest='doEndcap',       default=False, action='store_true')
    parser.add_option("--filesLim",       dest="filesLim",       default=10000, type=int)
    (options, args) = parser.parse_args()
    print(options)

    user = os.getcwd().split('/')[5]
    outfile_base = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/EmuValidation/'

    inlist_folders = []
    inlist_folders.append(outfile_base+"HHbbtautau_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+"/")

    print(inlist_folders)

    images_files = []
    posits_files = []
    shapes_files = []
    idscore_emu_files = []
    calibpt_emu_files = []

    for i in range(len(inlist_folders)):
        infolder = inlist_folders[i]
        
        if options.doBarrel:
            images_files.extend( glob.glob(infolder+'/barrel/CLTWimages_'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )
            posits_files.extend( glob.glob(infolder+'/barrel/CLTWpositions_'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )
            idscore_emu_files.extend( glob.glob(infolder+'/barrel/EMUidscore_'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )
            calibpt_emu_files.extend( glob.glob(infolder+'/barrel/EMUcalibpt_'+options.caloClNxM+'_CB_*.npz')[:options.filesLim] )

        if options.doEndcap:
            images_files.extend( glob.glob(infolder+'/endcap/CLTWimages_'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            posits_files.extend( glob.glob(infolder+'/endcap/CLTWpositions_'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            shapes_files.extend( glob.glob(infolder+'/endcap/CL3Dfeatures_cltw'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            idscore_emu_files.extend( glob.glob(infolder+'/endcap/EMUidscore_'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )
            calibpt_emu_files.extend( glob.glob(infolder+'/endcap/EMUcalibpt_'+options.caloClNxM+'_CE_*.npz')[:options.filesLim] )

    images_toConcat = []
    posits_toConcat = []
    shapes_toConcat = []
    idscore_emu_toConcat = []
    calibpt_emu_toConcat = []

    print('Total number of files:', len(images_files))
    for idx in range(len(images_files)):
        if idx%200==0: print('   ', idx)

        if options.doBarrel:
            images = np.load(images_files[idx], allow_pickle=True)['arr_0']
            posits = np.load(posits_files[idx], allow_pickle=True)['arr_0']
            idscore_emu = np.load(idscore_emu_files[idx], allow_pickle=True)['arr_0']
            calibpt_emu = np.load(calibpt_emu_files[idx], allow_pickle=True)['arr_0']

            images_toConcat.append(images)
            posits_toConcat.append(posits)
            idscore_emu_toConcat.append(idscore_emu)
            calibpt_emu_toConcat.append(calibpt_emu)

            del images, posits, idscore_emu, calibpt_emu

        if options.doEndcap:
            images = np.load(images_files[idx], allow_pickle=True)['arr_0']
            posits = np.load(posits_files[idx], allow_pickle=True)['arr_0']
            shapes = np.load(shapes_files[idx], allow_pickle=True)['arr_0']
            idscore_emu = np.load(idscore_emu_files[idx], allow_pickle=True)['arr_0']
            calibpt_emu = np.load(calibpt_emu_files[idx], allow_pickle=True)['arr_0']

            images_toConcat.append(images)
            posits_toConcat.append(posits)
            shapes_toConcat.append(shapes)
            idscore_emu_toConcat.append(idscore_emu)
            calibpt_emu_toConcat.append(calibpt_emu)

            del images, posits, idscore_emu, calibpt_emu


    if options.doBarrel:
        IMAGES = np.concatenate(images_toConcat)
        POSITS = np.concatenate(posits_toConcat)
        IDEMU = np.concatenate(idscore_emu_toConcat)
        PTEMU = np.concatenate(calibpt_emu_toConcat)

        del images_toConcat, posits_toConcat, shapes_toConcat, idscore_emu_toConcat, calibpt_emu_toConcat
        del images_files, posits_files, shapes_files, idscore_emu_files, calibpt_emu_files 

        print('shape IMAGES =', IMAGES.shape)
        print('shape POSITS =', POSITS.shape)
        print('shape IDEMU =', IDEMU.shape)
        print('shape PTEMU =', PTEMU.shape)

    if options.doEndcap:
        IMAGES = np.concatenate(images_toConcat)
        POSITS = np.concatenate(posits_toConcat)
        SHAPES = np.concatenate(shapes_toConcat)
        IDEMU = np.concatenate(idscore_emu_toConcat)
        PTEMU = np.concatenate(calibpt_emu_toConcat)

        del images_toConcat, posits_toConcat, shapes_toConcat, idscore_emu_toConcat, calibpt_emu_toConcat
        del images_files, posits_files, shapes_files, idscore_emu_files, calibpt_emu_files 

        print('shape IMAGES =', IMAGES.shape)
        print('shape POSITS =', POSITS.shape)
        print('shape SHAPES =', SHAPES.shape)
        print('shape IDEMU =',  IDEMU.shape)
        print('shape PTEMU =',  PTEMU.shape)


    if options.doBarrel:
        os.system('mkdir -p '+outfile_base+'/TauMinator_CB/tensors')

        np.savez_compressed(outfile_base+'/TauMinator_CB/tensors/images.npz', IMAGES)
        np.savez_compressed(outfile_base+'/TauMinator_CB/tensors/posits.npz', POSITS)
        np.savez_compressed(outfile_base+'/TauMinator_CB/tensors/idscore_emu.npz', IDEMU)
        np.savez_compressed(outfile_base+'/TauMinator_CB/tensors/calibpt_emu.npz', PTEMU)

    if options.doEndcap:
        os.system('mkdir -p '+outfile_base+'/TauMinator_CE/tensors')

        np.savez_compressed(outfile_base+'/TauMinator_CE/tensors/images.npz', IMAGES)
        np.savez_compressed(outfile_base+'/TauMinator_CE/tensors/posits.npz', POSITS)
        np.savez_compressed(outfile_base+'/TauMinator_CE/tensors/shapes.npz', SHAPES)
        np.savez_compressed(outfile_base+'/TauMinator_CE/tensors/idscore_emu.npz', IDEMU)
        np.savez_compressed(outfile_base+'/TauMinator_CE/tensors/calibpt_emu.npz', PTEMU)

