

import numpy as np
import matplotlib.pylab as plt

print(__name__)
#import numpy as np
#import matplotlib.pylab as plt

#main_tiles, extra_tiles = Nucleus.split_image(im, stitching='stitch_v1', verbose=False)
# verboseprint = print if verbose else lambda *a, **k: None

class Employee:
    def __init__(self, first, last, pay):
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+last
        
    def fullname(self):
        return('{} {}'.format(self.first, self.last))


class ImageTile:
    def __init__(self, img, coords, step):
        self.img = img
        self.coords = coords
        self.step = step
        self.INPUT_WIDTH, self.INPUT_HEIGHT = np.shape(img)[1], np.shape(img)[0]
     
    def show_me(self):
        print(f' This image tile has {self.INPUT_WIDTH} width and {self.INPUT_HEIGHT} height. The step is {self.step}.')
        plt.figure(figsize=(5,5))
        plt.imshow(self.img)


# class Splitters:
#     def __init__(self, img, coords=[0,0],):
#         """
#         Args:
#             img(ndarray): (H, W, 3)
#         """
#         self.img = img
#         self.INPUT_WIDTH, self.INPUT_HEIGHT = img.shape[1], img.shape[0]
#         self.coords = coords
        
        
#     def split_image(self, img, ):
#         """
#         Args:
              
#         """
#         main_coord =[]
#         main_tiles = []

#         for i in range(0,INPUT_HEIGHT,step):
#             for j in range(0,INPUT_WIDTH,step):
#                 verboseprint(f'Current image tile being generated is from {i} to {i+step} height (y) and {j} to {j+step} width (x).')
#                 main_tiles.append(img[i:i+step, j:j+step])
#                 main_coord.append([i,j])
#         return main_tiles, main_coord


#     def make_tiles(self, stitching='no_stitch', verbose=False):
        
#         if stitching='no_stitch':
#             split_image()
#         elif stitching='stitch_v1':
#             split_image()
#             print(
    
    
    
# step=128
# border_step=step

# #make MAIN TILES: squares step x step

# main_coord =[]
# main_tiles = []

# for i in range(0,INPUT_HEIGHT,step):
#     for j in range(0,INPUT_WIDTH,step):
#         print( "Current image is from "+ f'{i} to {i+step} height (y) and {j} to {j+step} width (x).')
#         main_tiles.append(im[i:i+step, j:j+step])
#         main_coord.append([i,j])
# print(np.shape(main_tiles))
# print(np.min(main_tiles))
# print(np.max(main_tiles))

# plt.figure()
# plt.imshow(main_tiles[1])


# # make vertical stripes to fix nuclei at borders: rectangles step x border_step
# v_borders = []
# v_coord =[]
# for i in range(0,INPUT_HEIGHT,step):
#     for j in range(0,INPUT_WIDTH,step):
#         if j!=0:
#             print( "Current v_border image is from "+ f'{i} to {i+step} height (y) and {j-int(border_step/2)} to {j+int(border_step/2)} width (x).')
#             v_borders.append(im[ i:i+step , j-int(border_step/2):j+int(border_step/2)])
#             v_coord.append([i,j-int(border_step/2)])

# print(np.shape(v_borders))
# print(np.min(v_borders))
# print(np.max(v_borders))

# plt.figure()
# plt.imshow(v_borders[0])


# # make horizontal stripes to fix nuclei at borders: rectangles border_step x border_step+step 
# h_borders = []
# h_coord =[]
# ys=range(0,INPUT_HEIGHT,step)
# xs=range(0,INPUT_WIDTH,step)

# for i in ys: # controls y
#     if i!=0:
#         for j in xs: # controls x
#             if j==0:
#                 print( "Current first h_border image is from "+ f'{i-int(border_step/2)} to {i+int(border_step/2)} height (y) and {j} to {j+step+int(border_step/2)} width (x).')
#                 h_borders.append(im[ i-int(border_step/2):i+int(border_step/2) , j:j+step+int(border_step/2)])
#                 h_coord.append([i-int(border_step/2),j])
#             elif j==xs[-1]:
#                 print( "Current last h_border image is from "+ f'{i-int(border_step/2)} to {i+int(border_step/2)} height (y) and {j-int(border_step/2)} to {j+step} width (x).')
#                 h_borders.append(im[ i-int(border_step/2):i+int(border_step/2) , j-int(border_step/2):j+step])
#                 h_coord.append([i-int(border_step/2),j-int(border_step/2)])
#             else:
#                 print( "Current h_border image is from "+ f'{i-int(border_step/2)} to {i+int(border_step/2)} height (y) and {j-int(border_step/2)} to {j+step+int(border_step/2)} width (x).')
#                 h_borders.append(im[ i-int(border_step/2):i+int(border_step/2) , j-int(border_step/2):j+step+int(border_step/2)])
#                 h_coord.append([i-int(border_step/2),j-int(border_step/2)])


# plt.figure()
# plt.imshow(h_borders[0])

# def main():
#     pass

# if __name__ == '__main__':
#     main()
    
