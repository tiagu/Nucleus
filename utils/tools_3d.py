# Nucleus 2022.04
# Tiago Rito, The Francis Crick Institute

import os
from skimage import io
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors
#!pip install Faker
from faker import Faker
fake = Faker()
import pandas as pd
import cv2 as cv2

#!pip install networkx==2.5
#!pip install cupy-cuda101
import multiprocessing as mp
from functools import partial
import time
import itertools
import networkx as nx
from scipy import ndimage
import plotly.express as px




def make_3d_mask(ROOT_PATH, input_img, top_range):
    #images_to_segment = os.listdir(ROOT_PATH+"/img/")
    #print(images_to_segment)
    #input_img=images_to_segment[0]
    #print(input_img)

    im = io.imread(ROOT_PATH+"img/"+input_img)
    print(im.shape)
    im = im[:,:,0:round(im.shape[2]/128)*128,0:round(im.shape[3]/128)*128]
    print(im.shape) #should be z-planes, channels, x,y

#     plt.figure(figsize=(15,5))
#     plt.imshow(np.flip(im[:,0,:,600],axis=0), cmap='Greys_r'); plt.axis('off')

#     plt.figure(figsize=(15,15))
#     plt.imshow(np.flip(im[50,0,:,:],axis=0), cmap='Greys_r'); plt.axis('off')


    # make mask matrix (relatively fast for ~150 z-planes)
    masks=[]
    for z in range(0,im.shape[0]):
        im = io.imread(ROOT_PATH+"masks/"+"z"+f"{z:03d}"+"z_"+input_img )
        masks.append(im)

    masks = np.array(masks)

    #make x x and y y
    masks = np.flipud(masks)
    print(masks.shape)

    # flip z if needed
    masks = masks[::-1, :, :]
    print(masks.shape)


    # clean masks matrix for very small masks (from overlays/ seg errors)
    z=[]
    mask_val=[]
    areas = []

    for i in range(masks.shape[0]):
        z_plane = np.asarray(masks[i,:,:])
        u, indices = np.unique(z_plane, return_counts=True)
        z.extend( np.repeat(i, len(u)-1) )
        mask_val.extend( np.delete(u,0) )
        areas.extend( np.delete(indices,0) )


    df = pd.DataFrame(list(zip(z, mask_val,areas)),columns=['z','masks_val','area'])
    
    plt.figure(figsize=(6,4))
    plt.hist(df.area,50, range=[0,top_range],facecolor='grey')
    
    return masks,df



    
def filter_and_save_3d_mask(masks, df, cutoff, ROOT_PATH, input_img):
#     ### CHOOSE CUTOFF to filter segmented small 2d areas #####
    df = df[df.area<cutoff]

    for i in range(masks.shape[0]):
        for j in list(df.masks_val[df.z==i]):
            masks[i,:,:] = np.where( np.asarray(masks[i,:,:])==j,
                                    0,
                                    masks[i,:,:])


    np.savez_compressed(ROOT_PATH+input_img+"_masks_merge_no_small.npz",masks)
    print("Saved npz with collection of 2D masks with small areas filtered out.")
    print(ROOT_PATH+input_img+"_masks_merge_no_small.npz")
    
    return masks








# QUALITY CONTROL
# from https://github.com/delestro/rand_cmap

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap










def graph_contructor(masks):
    
    global get_nucleus_v2
    
    def touch_z(masks_mat, z1, id1, z_dir):
        
        #get list of unique identifiers of next plane z_dir that touch z1_id1
        ind_xz,ind_yz = np.where( masks_mat[z1,:,:] == id1 )
        z_plane = masks_mat[z1+z_dir,ind_xz,ind_yz]
        z2 = np.unique(z_plane)
        z2 = z2[ z2!=0]
        z2 = list(z2) 

        # their percentage of overlap with z1_id1: #pixels of z2_id2 that overlap/ all #pixels z1_id1
        z2_perc=[]
        for _ in z2:
            z2_perc.append(round( 100*len(np.argwhere( z_plane == _ ))/ len(ind_xz)) )

        # gradient: all #pixels z2_id2/ all #pixels z1_id1
        z_grad=[]
        for _ in z2:
            z_grad.append( len(np.argwhere( masks_mat[z1+z_dir,:,:] == _ )) / len(ind_xz) )

        # jaccard
        z_jac=[]
        for _ in z2:
            a = list(np.argwhere( masks_mat[z1+z_dir,:,:] == _ )) # positions of test id
            b = list(np.argwhere( masks_mat[z1,:,:] == id1 ) ) # positions of id_now

            uni_z = a+b
            arr, uniq_cnt = np.unique(uni_z, axis=0, return_counts=True)
            #uniq_arr = arr[uniq_cnt==1]

            int_z = arr[uniq_cnt==2]

            jac = 1 - (len(int_z)/len(arr))

            z_jac.append( jac )


        return(z2, z2_perc, z_grad, z_jac)


    def get_nucleus_v2( z_now, id_now):
        edges_to_add=list()
        #go up if possible
        if z_now+1<masks.shape[0]:
            nu_high, nu_high_perc, grad_high, jac_high = touch_z(masks, z_now, id_now, z_dir=(+1) )
            if nu_high:
                for i in range(len(nu_high)):
                    if nu_high_perc[i]>30: #min overlap of 30%
                        if jac_high[i]==np.min(jac_high): # select one with lowest jaccard distance
                             edges_to_add.append([ str(z_now)+'z'+str(id_now),str(z_now+1)+'z'+str(nu_high[i]),
                                              {'weight': nu_high_perc[i], 'gradient_high':round(grad_high[i],3), 'jac_dist':round(jac_high[i],3) }])
        return(edges_to_add)
    
    
    #masks_mat = np.copy(masks)
    
    masks2 = np.copy(masks)
    
    df_to_work=[]
    for z in range(masks2.shape[0]):
        avail_nuc = np.unique(masks2[z,:,:])
        avail_nuc = list( avail_nuc[ avail_nuc!=0] )
        df_to_work.extend( [(z,x) for x in avail_nuc])
    
    
    pool = mp.Pool(30)
    to_add = pool.starmap(get_nucleus_v2, df_to_work)
    pool.close()
    
    return to_add



def graph_stats(G):
    #print("Length of graph")
    #print(len(G.nodes()))

    nuclei={0:[]}
    for i in nx.connected_components(G.to_undirected()):
        nuclei[ list(nuclei.keys())[-1]+1 ] = list(i)

    nuclei.pop(0)

    nodes_per_nucleus = {}  
    for i in nuclei.keys():
        nodes_per_nucleus[i]=len(list(nuclei[i]))

    print("Total number of nuclei in the stack (connected components):")
    print(len(nuclei.keys()))
    
    _ = plt.hist(nodes_per_nucleus.values(), 50, range=[0,50], facecolor='grey')
    print(np.median(list(nodes_per_nucleus.values())))
    
    return nuclei


def split_nuclei_jac_old(H, masks,i_nuc,graph_par_thres):
    H2 = H.copy()
    
    ord_grad = [ (node1, node2, data['jac_dist']) for node1, node2, data in H.edges(data=True) if 'jac_dist' in data]
    ord_grad = pd.DataFrame(ord_grad)
    ord_grad['z1'] = [int(x.split('z')[0]) for x in ord_grad[0]]
    ord_grad = ord_grad.sort_values(by=['z1'])
    ord_grad['present']=True
    
    for i in range(len(ord_grad)):
        if ord_grad[2][i]>0.7:   # remove automatically distances above .7
            H2.remove_edge( ord_grad[0][i], ord_grad[1][i] )
            ord_grad.loc[ (ord_grad[0]==ord_grad[0][i]) & (ord_grad[1]==ord_grad[1][i]), ('present')   ] = False
    
    nuclei_small={0:[]}
    for i in nx.connected_components(H2.to_undirected()):
        nuclei_small[ list(nuclei_small.keys())[-1]+1 ] = list(i)

    nuclei_small.pop(0)
    
    for i in nuclei_small.keys():
        if len(nuclei_small[i])> (graph_par_thres+2): #threshold set of +2 z
            
            df_extra = ord_grad[ np.asarray(ord_grad[0].isin(nuclei_small[i])) | np.asarray(ord_grad[1].isin(nuclei_small[i] )) ]
            df_extra = df_extra[df_extra.present==True ]

            df_extra.loc[ df_extra.z1==np.min(df_extra.z1), 'present'] = False
            df_extra.loc[ df_extra.z1==np.max(df_extra.z1), 'present'] = False

            df_extra = df_extra[df_extra.present==True ]

            if np.max(df_extra[2])<0.2: #this is too low to make sense and prob Jac dist will not help
                next
            else:
                rand_pos= np.random.choice(np.where(df_extra[2] == np.max(df_extra[2]))[0]) # if value is the same per chance choose random

                node1= ord_grad[0][ df_extra.index[ rand_pos] ]
                node2= ord_grad[1][ df_extra.index[ rand_pos] ]
                H2.remove_edge( node1, node2)
                ord_grad.loc[ (ord_grad[0]==node1) & (ord_grad[1]==node2), ('present')   ] = False

    labels_nu1={}
    ch=0
    for i in nx.connected_components(H2.to_undirected()):
        ch=ch+1
        labels_nu1.setdefault(ch,[]).extend( list(i) )


    return labels_nu1
            
            
            
            
            
            
def split_nuclei_jac(H, masks,i_nuc,graph_par_thres):
    
    #print(i_nuc)

    H2 = H.copy()

    ord_grad = [ (node1, node2, data['jac_dist']) for node1, node2, data in H.edges(data=True) if 'jac_dist' in data]
    ord_grad = pd.DataFrame(ord_grad)
    ord_grad['z1'] = [int(x.split('z')[0]) for x in ord_grad[0]]
    ord_grad = ord_grad.sort_values(by=['z1'])
    ord_grad['present']=True

    #print(ord_grad)

    # distances too large as a first pass
    for i in range(len(ord_grad)):
        if ord_grad[2][i]>0.7:   # remove automatically distances above .7
            H2.remove_edge( ord_grad[0][i], ord_grad[1][i] )
            ord_grad.loc[ (ord_grad[0]==ord_grad[0][i]) & (ord_grad[1]==ord_grad[1][i]), ('present')   ] = False

    nuclei_small={0:[]}
    for i in nx.connected_components(H2.to_undirected()):
        nuclei_small[ list(nuclei_small.keys())[-1]+1 ] = list(i)
    nuclei_small.pop(0)

    
    #inspect smaller subgraphs for nuclei coexisting in same z
    for i in nuclei_small.keys():
        if len(nuclei_small[i])> (graph_par_thres): #if there are still big models
            df_extra = ord_grad[ np.asarray(ord_grad[0].isin(nuclei_small[i])) | np.asarray(ord_grad[1].isin(nuclei_small[i] )) ]
            df_extra = df_extra[df_extra.present==True ]

            df_extra.loc[ df_extra.z1==np.min(df_extra.z1), 'present'] = False
            df_extra.loc[ df_extra.z1==np.max(df_extra.z1), 'present'] = False

            df_extra = df_extra[df_extra.present==True ]

            c_df = np.unique(df_extra.z1,return_counts=True)

            if np.any(c_df[1]>=2): # check if it is a multiple nuclei in one z model.
                # identify junctions
                junctions = []
                for i in range(len(c_df[1])-1):
                    if c_df[1][i]==c_df[1][i+1]:
                        junctions.append(0)
                    else: junctions.append(1)
                junctions = np.asarray(junctions)

                # if duplication in same z then cut the edge with highest jac distance
                for j in range(len(junctions)):
                    if junctions[j]==1:
                        _ = df_extra[ df_extra.z1.isin([c_df[0][j],c_df[0][j+1]]) ] 
                        if _[0].duplicated().any():
                            for i in _[0].index[_[0].duplicated()]:
                                node1 = _[0][ _[0]==_[0][i] ][ _[2][ _[0]==_[0][i] ] == np.max(_[2][ _[0]==_[0][i] ] ) ].values[0]
                                node2 = _[1][ _[0]==_[0][i] ][ _[2][ _[0]==_[0][i] ] == np.max(_[2][ _[0]==_[0][i] ] ) ].values[0]
                                if H2.has_edge(node1, node2): H2.remove_edge( node1, node2)
                                df_extra.loc[ (df_extra[0]==node1) & (df_extra[1]==node2), ('present')   ] = False
                                #print("left dup")
                                #print("removed edges between: "+node1+"  "+node2)
                        elif _[1].duplicated().any():
                            for i in _[1].index[_[1].duplicated()]:
                                node1 = _[0][ _[1]==_[1][i] ][ _[2][ _[1]==_[1][i] ] == np.max(_[2][ _[1]==_[1][i] ] ) ].values[0]
                                node2 = _[1][ _[1]==_[1][i] ][ _[2][ _[1]==_[1][i] ] == np.max(_[2][ _[1]==_[1][i] ] ) ].values[0]
                                if H2.has_edge(node1, node2): H2.remove_edge( node1, node2)
                                df_extra.loc[ (df_extra[0]==node1) & (df_extra[1]==node2), ('present')   ] = False
                                #print("right dup")
                                #print("removed edges between: "+node1+"  "+node2)


    nuclei_small={0:[]}
    for i in nx.connected_components(H2.to_undirected()):
        nuclei_small[ list(nuclei_small.keys())[-1]+1 ] = list(i)
    nuclei_small.pop(0) 

    #print(nuclei_small)

    #inspect further the smaller subgraphs
    for i in nuclei_small.keys():
        if len(nuclei_small[i])> (graph_par_thres): #if there are still big models

            df_extra = ord_grad[ np.asarray(ord_grad[0].isin(nuclei_small[i])) | np.asarray(ord_grad[1].isin(nuclei_small[i] )) ]
            df_extra = df_extra[df_extra.present==True ]

            df_extra.loc[ df_extra.z1==np.min(df_extra.z1), 'present'] = False
            df_extra.loc[ df_extra.z1==np.max(df_extra.z1), 'present'] = False

            df_extra = df_extra[df_extra.present==True ]

            if np.max(df_extra[2])<0.2: #this is too low to make sense and prob Jac dist will not help
                next
            else:
                rand_pos= np.random.choice(np.where(df_extra[2] == np.max(df_extra[2]))[0]) # if value is the same per chance choose random

                node1= ord_grad[0][ df_extra.index[ rand_pos] ]
                node2= ord_grad[1][ df_extra.index[ rand_pos] ]
                if H2.has_edge(node1, node2): H2.remove_edge( node1, node2)
                ord_grad.loc[ (ord_grad[0]==node1) & (ord_grad[1]==node2), ('present')   ] = False
    
    labels_nu1={}
    ch=0
    for i in nx.connected_components(H2.to_undirected()):
        ch=ch+1
        labels_nu1.setdefault(ch,[]).extend( list(i) )


    return labels_nu1 







def consolidate(G, masks, nuclei, graph_par_thres , hard_cutoff):
    print("Total number of nuclei in the stack:")
    print(len(nuclei.keys()))
    
    nodes_per_nucleus = {}
    for i in nuclei.keys():
        nodes_per_nucleus[i]=len(list(nuclei[i]))
    
    graph_par_thres = graph_par_thres
    #define which nuclei to refine (split)
    nu_models = [i for i in nodes_per_nucleus.keys() if nodes_per_nucleus[i]>graph_par_thres]
    print("Nuclei to split:")
    print(len(nu_models))

    w=0
    w_total=len(nu_models)
    for i in nu_models:

        # split nuclei
        H = G.subgraph(nuclei[i])

        #labels_nu1 = split_nuclei_func(H, masks,i)
        labels_nu1 = split_nuclei_jac(H, masks,i, graph_par_thres)

        nuclei.pop(i)

        #replaces the original unsplit entry with split nuclei
        for k in labels_nu1.keys():
            nuclei[ str(i)+"_"+str(k) ] = labels_nu1[k]

        #w=w+1
        #if round(w*100/w_total)% 25 == 0:
            #print(str(round(w*100/w_total)),end=' .. ')


    print("Total number of nuclei in the stack:")
    print(len(nuclei.keys()))
    
    
    nodes_per_nucleus = {}
    for i in nuclei.keys():
        nodes_per_nucleus[i]=len(list(nuclei[i]))

    #define which nuclei to refine (split)
    nu_models = [i for i in nodes_per_nucleus.keys() if nodes_per_nucleus[i]>graph_par_thres+hard_cutoff]
    print("Nuclei to further split")
    print(len(nu_models))
    print("\n")
    
    #generous +10 above which there can be no nucleus
    nu_models_old=0

    while ((len(nu_models)>0) and (not(len(nu_models)==nu_models_old))):

        nu_models_old=len(nu_models)

        w=0
        w_total=len(nu_models)
        for i in nu_models:

            # split nuclei
            H = G.subgraph(nuclei[i])

            #labels_nu1 = split_nuclei_func(H, masks,i)
            labels_nu1 = split_nuclei_jac(H, masks,i,graph_par_thres)

            nuclei.pop(i)

            #replaces the original unsplit entry with split nuclei
            for k in labels_nu1.keys():
                nuclei[ str(i)+"_"+str(k) ] = labels_nu1[k]

            w=w+1
            #if round(w*100/w_total)% 25 == 0:
            #    print(str(round(w*100/w_total)),end=' .. ')


        nodes_per_nucleus = {}
        for i in nuclei.keys():
            nodes_per_nucleus[i]=len(list(nuclei[i]))

        #define which nuclei to refine (split)
        nu_models = [i for i in nodes_per_nucleus.keys() if nodes_per_nucleus[i]>graph_par_thres+hard_cutoff]
        print("Nuclei to further split")
        print(len(nu_models))
        print("\n")
        
        
    print("\n")
    print("Total number of nuclei in the stack:")
    print(len(nuclei.keys()))   

    print("\n")

    return nuclei






import cupy as cp



def make_3d_mat(nuclei, masks):
    
    id_cols_z=[]
    id_cols_id=[]

    for i in nuclei.keys():
        nu1 = [(np.int64(x[0]),np.int64(x[1])) for x in [x.split('z') for x in nuclei[i]] ]
        nu1 = list(zip(*nu1))
        id_cols_z.append( list(nu1[0])) 
        id_cols_id.append( list( nu1[1])) 

    n = len(max(id_cols_z, key=len))
    lst_2 = [x + [cp.nan]*(n-len(x)) for x in id_cols_z]
    id_cols_z = np.array(lst_2)


    n = len(max(id_cols_id, key=len))
    lst_2 = [x + [np.nan]*(n-len(x)) for x in id_cols_id]
    id_cols_id = np.array(lst_2)


    masks_mat = cp.asarray(masks)
    masks_mat_3d = cp.zeros(masks_mat.shape, dtype='uint64')

    nucle=cp.asarray(0)

    nuclei_dic_keys= cp.asarray(np.uint64(list(nuclei.keys())))
    cols_z = cp.asarray(np.asarray(id_cols_z, dtype='uint64'))
    cols_id = cp.asarray(np.asarray(id_cols_id, dtype='uint64'))

    cp.cuda.Stream.null.synchronize()

    for i in range(len(nuclei_dic_keys)): #which nuclei
        nucle = nucle+1
        cols_z_nuc = cols_z[i]
        cols_id_nuc = cols_id[i]
        
        #key_to_mask[ nuclei_dic_keys[i] ] = nucle
        
        for j in range(len(cols_z_nuc)):
            if cols_z_nuc[j]==cp.nan:
                break
            else:
                masks_mat_3d[cols_z_nuc[j],:,:] = cp.where( masks_mat[cols_z_nuc[j],:,:] == cols_id_nuc[j], nucle, masks_mat_3d[cols_z_nuc[j],:,:]  )

    return masks_mat_3d
          




def get_feature_table(input_img, im, masks_3d):
    
    global get_nu_stats
    input_img = input_img

    nus = np.unique(masks_3d)
    nus = np.delete(nus, 0)


    def get_nu_stats(i):

        nu1 = np.where(masks_3d==i)

        #nuclear averages
        nuclear_avgs = []
        for h in range(im.shape[1]):
            nuclear_avgs.append(  round(np.mean(im[:,h,:,:][nu1]),3)   )


        # cell averages
        nu_3d = masks_3d.copy()

        trim_z_min=np.min(nu1[0])-2
        if trim_z_min<0:
            trim_z_min=0

        trim_z_max=np.max(nu1[0])+2
        if trim_z_max>masks_3d.shape[0]:
            trim_z_max=masks_3d.shape[0]

        trim_x_min=np.min(nu1[1])-10
        if trim_x_min<0:
            trim_x_min=0

        trim_x_max=np.max(nu1[1])+10
        if trim_x_max>masks_3d.shape[1]:
            trim_x_max=masks_3d.shape[1]

        trim_y_min=np.min(nu1[2])-10
        if trim_y_min<0:
            trim_y_min=0

        trim_y_max=np.max(nu1[2])+10
        if trim_y_max>masks_3d.shape[2]:
            trim_y_max=masks_3d.shape[2]

        nu_3d = nu_3d[trim_z_min:trim_z_max, trim_x_min:trim_x_max, trim_y_min:trim_y_max]
        nu_3d = np.where( nu_3d==i, 1,0 ) #make binary




        opening = ndimage.binary_dilation(input=nu_3d, iterations=3)

        nu1_dilated = list(np.where(opening==1))

        nu1_dilated[0]= nu1_dilated[0] + trim_z_min
        nu1_dilated[1]= nu1_dilated[1] + trim_x_min
        nu1_dilated[2]= nu1_dilated[2] + trim_y_min

        nu1_dilated=tuple(nu1_dilated)

        hood_avgs = []
        for h in range(im.shape[1]):
            hood_avgs.append(  round(np.mean(im[:,h,:,:][nu1_dilated]),3)   )


        # cell averages
        cyto = opening-nu_3d

        nu1_cyto = list(np.where(cyto==1))

        nu1_cyto[0]= nu1_cyto[0] + trim_z_min
        nu1_cyto[1]= nu1_cyto[1] + trim_x_min
        nu1_cyto[2]= nu1_cyto[2] + trim_y_min

        nu1_cyto=tuple(nu1_cyto)

        cyto_avgs = []
        for h in range(im.shape[1]):
            cyto_avgs.append(  round(np.mean(im[:,h,:,:][nu1_cyto]),3)   )



        #centroid
        mass_c = ndimage.measurements.center_of_mass(nu_3d)
        mass_c = round(mass_c[0]) + trim_z_min , round(mass_c[1])+ trim_x_min, round(mass_c[2])+ trim_y_min


        #volume
        vol_nu1=len(nu1[0])


        return( (input_img, i, vol_nu1, mass_c, nuclear_avgs, hood_avgs, cyto_avgs) )


    pool = mp.Pool(15)
    result = pool.map(get_nu_stats, nus)
    
    return result
    


    



# plot a specific nucleus instance
def plot_nucleus_model(masks_3d, i):
    # i=159
    nu1 = np.where(masks_3d==i)
    len(nu1[0])
    
    nu_3d = masks_3d.copy()
    
    trim_z_min=np.min(nu1[0])-2
    if trim_z_min<0:
        trim_z_min=0
    
    trim_z_max=np.max(nu1[0])+2
    if trim_z_max>masks_3d.shape[0]:
        trim_z_max=masks_3d.shape[0]
    
    trim_x_min=np.min(nu1[1])-10
    if trim_x_min<0:
        trim_x_min=0
    
    trim_x_max=np.max(nu1[1])+10
    if trim_x_max>masks_3d.shape[1]:
        trim_x_max=masks_3d.shape[1]
    
    trim_y_min=np.min(nu1[2])-10
    if trim_y_min<0:
        trim_y_min=0
    
    trim_y_max=np.max(nu1[2])+10
    if trim_y_max>masks_3d.shape[2]:
        trim_y_max=masks_3d.shape[2]
    
    nu_3d = nu_3d[trim_z_min:trim_z_max, trim_x_min:trim_x_max, trim_y_min:trim_y_max]
    nu_3d = np.where( nu_3d==i, 1,0 ) #make binary
    
    mass_c = ndimage.measurements.center_of_mass(nu_3d)
    print(mass_c)
    
    z,x,y = np.where(nu_3d!=0)
    df2 = pd.DataFrame({"x":x,"y":y,"z":z, 'color':nu_3d[z,x,y]})
    df2
    
    df2.color = [str(i) for i in df2.color]
    
    fig = px.scatter_3d(df2, x=x, y=y, z=z, 
                        color='color', opacity=1,
                       )
    fig.show()




# overlay mask and channel



from skimage.io import imsave

# from https://www.javaer101.com/en/article/17121494.html
def replace_with_dict2(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]



def export_3d_array_rgb(masks_3d_array, filename):

    #masks_3d_array = masks_toy_3d.copy()    

    s_array=masks_3d_array.shape
    print(s_array)

    keys= np.unique(masks_3d_array)

    print("\nUnique nuclei ids in image:")
    print(len(keys))

    keys = np.append(keys, np.max(keys)+1 )

    cols = np.asarray([fake.unique.rgb_color() for _ in range(len(keys))]) # limited to 256 colors including black and white
    #cols = np.asarray([str(random.choice(range(0,255)))+','+ str(random.choice(range(0,255)))+',' + str(random.choice(range(0,255)))
                      # for _ in range(len(keys))])
    
    #throw dice again on repeated colors
#     rep_cols=len(cols)-len(np.unique(cols))
#     _, idx = np.unique(cols, return_index=True)
#     cols = cols[np.sort(idx)]
#     for _ in range(rep_cols):
#         np.append(cols, fake.rgb_color() )

    #remove blacks
    ind_black = [i for i in range(len(cols)) if cols[i] == '0, 0, 0']
    np.put(cols, 
           ind_black, 
           [fake.rgb_color() for _ in range(len(ind_black))]
           )

    cols = np.insert(cols, 0, '0, 0, 0')
    
    into_rgb = dict(zip(keys,cols))

    masks_3d_array_rgb = replace_with_dict2(masks_3d_array, into_rgb)

    r=np.zeros(masks_3d_array_rgb.shape)
    g=np.zeros(masks_3d_array_rgb.shape)
    b=np.zeros(masks_3d_array_rgb.shape)

    for z in range(masks_3d_array_rgb.shape[0]):
        for x in range(masks_3d_array_rgb.shape[1]):
            for y in range(masks_3d_array_rgb.shape[2]):
                r[z,x,y] = masks_3d_array_rgb[z,x,y].split(',')[0]
                g[z,x,y] = masks_3d_array_rgb[z,x,y].split(',')[1]
                b[z,x,y] = masks_3d_array_rgb[z,x,y].split(',')[2]

    rgb = np.stack((r, g, b), axis=3)
    

    imsave(filename, rgb, bigtiff=True)
    print("dimensions: "+str(rgb.shape))
    print("exported rgb tif file to "+filename)
    print("done!")
    



if __name__ == '__main__':
     main()


