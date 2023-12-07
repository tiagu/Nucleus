import numpy as np
import networkx as nx
import multiprocessing as mp
from functools import partial
import time
from multiprocessing import Array




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








def get_nucleus_v2(z_now, id_now):
    edges_to_add=list()
    #go up if possible
    if z_now+1<masks_mat.shape[0]:
        nu_high, nu_high_perc, grad_high, jac_high = touch_z(masks_mat, z_now, id_now, z_dir=(+1) )
        if nu_high:
            for i in range(len(nu_high)):
                if nu_high_perc[i]>30: #min overlap of 30%
                    if jac_high[i]==np.min(jac_high): # select one with lowest jaccard distance
                         edges_to_add.append([ str(z_now)+'z'+str(id_now),str(z_now+1)+'z'+str(nu_high[i]),{'weight': nu_high_perc[i], 'gradient_high':round(grad_high[i],3), 'jac_dist':round(jac_high[i],3) }])
    return(edges_to_add)







def main(ROOT_PATH, input_img, masks):
    
    tic = time.perf_counter()
    
    masks_shape=masks.shape
    
    
    X = RawArray('d', masks_shape[0] * masks_shape[1])
    X_np = np.frombuffer(X).reshape(X_shape)
    np.copyto(X_np, masks_mat)
    
    
    #masks_mat = np.copy(masks)
    
    df_to_work=[]
    for z in range(masks_mat.shape[0]):
        avail_nuc = np.unique(masks_mat[z,:,:])
        avail_nuc = list( avail_nuc[ avail_nuc!=0] )
        df_to_work.extend( [(z,x) for x in avail_nuc])


    pool = mp.Pool(30, initializer=init_worker, initargs=(X, X_shape))
    to_add = pool.starmap(get_nucleus_v2, df_to_work)
    pool.close()

    G = nx.DiGraph()
    to_add2=[tuple(x[0]) for x in to_add if not x==[]]
    G.add_edges_from(to_add2)

    nx.write_gpickle(G, ROOT_PATH+input_img+"_G.gpickle")

    toc = time.perf_counter()
    print(f"Done nuclei connection graph in {toc - tic:0.4f} seconds")
    



if __name__ == '__main__':
    main()

    
    
    
    
    from sys import stdin
from multiprocessing import Pool, Array, Process

def count_it( key ):
  count = 0
  for c in toShare:
    if c == key:
      count += 1
  return count

if __name__ == '__main__':
  # allocate shared array - want lock=False in this case since we 
  # aren't writing to it and want to allow multiple processes to access
  # at the same time - I think with lock=True there would be little or 
  # no speedup
  maxLength = 50
  toShare = Array('c', maxLength, lock=False)

  # fork
  pool = Pool()

  # can set data after fork
  testData = "abcabcs bsdfsdf gdfg dffdgdfg sdfsdfsd sdfdsfsdf"
  if len(testData) > maxLength:
      raise ValueError, "Shared array too small to hold data"
  toShare[:len(testData)] = testData

  print pool.map( count_it, ["a", "b", "s", "d"] )
