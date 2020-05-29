import numpy as np
import matplotlib.pylab as plt
import cv2
import torch
from pycocotools.coco import COCO
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

print(__name__)


VERBOSE=False


if VERBOSE:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    verboseprint = lambda *a, **k: None 




class ImageTile:
    def __init__(self, img, coords, step):
        self.img = img
        self.coords = coords
        self.step = step
        self.INPUT_WIDTH, self.INPUT_HEIGHT = np.shape(img)[1], np.shape(img)[0]
     
    def show_me(self):
        print(f' This image tile goes from {self.coords[1]} to  {self.coords[1]+self.INPUT_WIDTH} width (x) and from {self.coords[0]} to {self.coords[0]+self.INPUT_HEIGHT} height (y). The step is {self.step}.')
        plt.figure(figsize=(5,5))
        plt.imshow(self.img)



    
class ImageInput:
    def __init__(self, img_str, coords = None, step=None):
        
        self.img = cv2.imread(img_str)
        self.img = 255*((self.img - np.min(self.img))/np.ptp(self.img)) # between 0-255
        self.img = np.uint8(self.img)
        
        self.INPUT_HEIGHT=np.shape(self.img)[0]
        self.INPUT_WIDTH=np.shape(self.img)[1]
        
        if coords is None:
            self.coords = [0,0]
        else:
            self.coords = coords
        
        if step is None:
            self.step=256
        else:
            self.step = step
    
    def show_me(self):
        print(f'Input image shape: {np.shape(self.img)}') 
        print(f'Minimum pixel value: {np.min(self.img)}')
        print(f'Maximum pixel value: {np.max(self.img)}')
        plt.figure(figsize=(5,5))
        plt.imshow(self.img)

    def split_image(self):
        main_tiles = []
        for i in range(0,self.INPUT_HEIGHT, self.step):
            for j in range(0,self.INPUT_WIDTH, self.step):
                verboseprint(f'Current image tile being generated is from {i} to {i+self.step} height (y) and {j} to {j+self.step} width (x).')
                main_tiles.append( ImageTile(self.img[i:i+self.step, j:j+self.step], [i,j], self.step ) )
        return main_tiles

    def split_vertical(self): # make vertical stripes to fix nuclei at borders: rectangles step x step
        v_borders = []
        for i in range(0,self.INPUT_HEIGHT,self.step):
            for j in range(0,self.INPUT_WIDTH,self.step):
                if j!=0:
                    verboseprint( "Current v_border image is from "+ f'{i} to {i+self.step} height (y) and {j-int(self.step/2)} to {j+int(self.step/2)} width (x).')
                    v_borders.append(ImageTile(self.img[ i:i+self.step , j-int(self.step/2):j+int(self.step/2)], [i,j-int(self.step/2)], self.step )    )
        return v_borders


    def split_horizontal(self): # make horizontal stripes to fix nuclei at borders: rectangles step x border_step+step 
        h_borders = []
        ys=range(0,self.INPUT_HEIGHT,self.step)
        xs=range(0,self.INPUT_WIDTH,self.step)
        for i in ys: # controls y
            if i!=0:
                for j in xs: # controls x
                    if j==0:
                        verboseprint( "Current first h_border image is from "+ f'{i-int(self.step/2)} to {i+int(self.step/2)} height (y) and {j} to {j+self.step+int(self.step/2)} width (x).')
                        h_borders.append(ImageTile(self.img[ i-int(self.step/2):i+int(self.step/2) , j:j+self.step+int(self.step/2)], [i-int(self.step/2),j], self.step))
                    elif j==xs[-1]:
                        verboseprint( "Current last h_border image is from "+ f'{i-int(self.step/2)} to {i+int(self.step/2)} height (y) and {j-int(self.step/2)} to {j+self.step} width (x).')
                        h_borders.append(ImageTile(self.img[ i-int(self.step/2):i+int(self.step/2) , j-int(self.step/2):j+self.step], [i-int(self.step/2),j-int(self.step/2)],self.step ))
                    else:
                        verboseprint( "Current h_border image is from "+ f'{i-int(self.step/2)} to {i+int(self.step/2)} height (y) and {j-int(self.step/2)} to {j+self.step+int(self.step/2)} width (x).')
                        h_borders.append(ImageTile(self.img[ i-int(self.step/2):i+int(self.step/2) , j-int(self.step/2):j+self.step+int(self.step/2)], [i-int(self.step/2),j-int(self.step/2)], self.step))
        return h_borders

    def make_tiles(self, how):
        if how=='simple':
            print(f'Using step of {self.step}px.')
            print(f'Splitting image in main tiles...')
            main_tiles = self.split_image()
            return main_tiles

        elif how=='stitch_v1':
            print(f'Using step of {self.step}px.')
            print(f'Splitting image in main tiles...')
            main_tiles = self.split_image()
            print(f'Splitting image in vertical border tiles...')
            vertical_tiles = self.split_vertical()
            print(f'Splitting image in horizontal border tiles...')
            horizontal_tiles = self.split_horizontal()
            return main_tiles, vertical_tiles, horizontal_tiles

        elif how=='stitch_v2':
            print("TBI...")

        elif how=='stitch_polygons':
            print("TBI...")

        else:
            print("Ups! Nothing to do here.")









class Stitcher:    
    def __init__(self, input_img):
        
        self.INPUT_HEIGHT= input_img.INPUT_HEIGHT
        self.INPUT_WIDTH = input_img.INPUT_WIDTH
        self.step   = input_img.step

        self.nuclei_tally = 1
        
    def no_stitch(self, tiles_col=None ,instances_col=None): # no stitiching of instances

        if (tiles_col is None) or (len(tiles_col)>1):
            print("Need just a list of the main image tiles.")
        elif len(tiles_col)==1:
            pass
        else:
            print("Error!")

        if (instances_col is None) or (len(instances_col)>1):
            print("Need just a list of instances for the main image tiles.")
        elif len(instances_col)==1:
            pass
        else:
            print("Error!")
        
        
        seg_mask_no_stitch = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH),dtype=torch.int32)

        m_tiles = tiles_col[0]
        m_out = instances_col[0]
        verboseprint(f'The number of main tiles in this image is {len(m_out)}')
        for i in range(0,len(m_out)):
           a=m_out[i].pred_masks
           verboseprint(f'The number of instances in this tile is: {len(a)}')
           for nucleus in range(0, len(a)):
               x,y = torch.where(a[nucleus]==1)
               seg_mask_no_stitch[x+m_tiles[i].coords[0] , y + m_tiles[i].coords[1]] = self.nuclei_tally # nucleus coord acquires its unique tally number
               self.nuclei_tally += 1

        return seg_mask_no_stitch.cpu().numpy(), self.nuclei_tally


    def stitch_v1(self, tiles_col=None ,instances_col=None, margin=5): # first version of stitiching of instances: an imperfect over-kill!

        if (tiles_col is None) or (len(tiles_col)<3) or (len(tiles_col)>3):
            print("Need lists of the main, vertical and horizontal image tiles.")
        elif len(tiles_col)==3:
            m_tiles, v_tiles, h_tiles = tiles_col
        else:
            print("Error!")

        if (instances_col is None) or (len(instances_col)<3) or (len(instances_col)>3):
            print("Need lists of instances for the main, vertical and horizontal tiles.")
        elif len(instances_col)==3:
            m_out, v_out, h_out = instances_col
        else:
            print("Error!")
        
        seg_mask = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH),dtype=torch.int32)

        # select only good instances from main tiles
        verboseprint(f'The number of main tiles in this image is {len(m_tiles)}')

        stitch_borders = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH))
        for i in range(0,self.INPUT_HEIGHT,self.step):
                for j in range(0,self.INPUT_WIDTH,self.step):
                    stitch_borders[i:i+self.step, j-margin:j+margin]=1
                    stitch_borders[i-margin:i+margin, j:j+self.step]=1
                
        for i in range(0,len(m_out)):
            a=m_out[i].pred_masks
            for nucleus in range(0, len(a)):
                x,y = torch.where(a[nucleus]==1)
                if torch.max(stitch_borders[x+m_tiles[i].coords[0] , y + m_tiles[i].coords[1]])==torch.tensor(0):
                    seg_mask[x+m_tiles[i].coords[0] , y + m_tiles[i].coords[1]] = self.nuclei_tally
                    self.nuclei_tally += 1

        # add good instances from vertical tiles
        verboseprint(f'The number of vertical tiles in this image is {len(v_tiles)}')

        stitch_borders_v = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH))
        for i in range(0,self.INPUT_HEIGHT,self.step):
            for j in range(0,self.INPUT_WIDTH,self.step):
                stitch_borders_v[i:i+self.step, j-margin:j+margin]=1

        stitch_borders_h = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH))
        for i in range(0,self.INPUT_HEIGHT,self.step):
            for j in range(0,self.INPUT_WIDTH,self.step):
                if i!=0:
                    stitch_borders_h[i-margin:i+margin, j:j+self.step]=1


        for i in range(0,len(v_out)):
            a=v_out[i].pred_masks
            for nucleus in range(0, len(a)):
                x,y = torch.where(a[nucleus]==1)
                cond1=torch.max(stitch_borders_v[x+v_tiles[i].coords[0] , y + v_tiles[i].coords[1]])==torch.tensor(1) #on the v_border
                cond2=torch.max(stitch_borders_h[x+v_tiles[i].coords[0] , y + v_tiles[i].coords[1]])==torch.tensor(0) #not on the horizontal
                if cond1 and cond2:
                    seg_mask[x+v_tiles[i].coords[0] , y + v_tiles[i].coords[1]] = self.nuclei_tally
                    self.nuclei_tally += 1   


        # add good instances from horizontal tiles with attention to corners, i.e. minimize bad instances whr 4 tiles meet
        verboseprint(f'The number of horizontal tiles in this image is {len(h_tiles)}')
        for i in range(0,len(h_out)):
            a=h_out[i].pred_masks
            if h_tiles[i].coords[1]== 0: # left edge of image case
                stitch_borders_x = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH))
                i_yy=h_tiles[i].coords[0]+int(self.step/2)
                stitch_borders_x[i_yy-margin:i_yy+margin ,  h_tiles[i].coords[1]:h_tiles[i].coords[1]+self.step]=1 # leaves out the +int(self.step/2) 
                for nucleus in range(0, len(a)):
                    x,y = torch.where(a[nucleus]==1)
                    cond1=torch.max(stitch_borders_x[x+h_tiles[i].coords[0] , y + h_tiles[i].coords[1]])==torch.tensor(1) #on the h_border
                    if cond1:
                        seg_mask[x+h_tiles[i].coords[0] , y + h_tiles[i].coords[1]] = self.nuclei_tally
                        self.nuclei_tally += 1

            elif (h_tiles[i].coords[1]+self.step+int(self.step/2))== self.INPUT_WIDTH: # right edge of image case
                stitch_borders_x = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH))
                i_yy=h_tiles[i].coords[0]+int(self.step/2)
                stitch_borders_x[i_yy-margin:i_yy+margin ,  h_tiles[i].coords[1]+int(self.step/2):h_tiles[i].coords[1]+self.step+int(self.step/2)]=1 # leaves out the int(self.step/2) 
                for nucleus in range(0, len(a)):
                    x,y = torch.where(a[nucleus]==1)
                    cond1=torch.max(stitch_borders_x[x+h_tiles[i].coords[0] , y + h_tiles[i].coords[1]])==torch.tensor(1) 
                    if cond1:
                        seg_mask[x+h_tiles[i].coords[0] , y + h_tiles[i].coords[1]] = self.nuclei_tally
                        self.nuclei_tally += 1

            else: # in the middle images
                stitch_borders_z = torch.zeros((self.INPUT_HEIGHT,self.INPUT_WIDTH))
                i_yy=h_tiles[i].coords[0]+int(self.step/2)
                stitch_borders_z[i_yy-margin:i_yy+margin ,  h_tiles[i].coords[1]+int(self.step/2) : h_tiles[i].coords[1]+self.step+int(self.step/2)  ]=1 # leaves out the int(self.step/2) 
                for nucleus in range(0, len(a)):
                    x,y = torch.where(a[nucleus]==1)
                    cond1=torch.max(stitch_borders_z[x+h_tiles[i].coords[0] , y + h_tiles[i].coords[1]])==torch.tensor(1) 
                    if cond1:
                        seg_mask[x+h_tiles[i].coords[0] , y + h_tiles[i].coords[1]] = self.nuclei_tally
                        self.nuclei_tally += 1
        

        return seg_mask.cpu().numpy(), self.nuclei_tally






# slightly modified version of pycocotools showAnns to display many instances of same type
class coco_nucleus(COCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)
        
    def showAnns_Nucleus(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                            color.append(c)
                    else:
                        # mask
                        print("Mask")
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.2)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1.5, alpha=0.8)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])




if __name__ == '__main__':
     main()