import numpy as np
import matplotlib.pylab as plt
import cv2
from skimage import measure
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
from pycocotools import mask
import glob
from tqdm import tqdm_notebook
from multiprocessing import Pool
from multiprocessing import cpu_count
import timeit

from functools import partial



print(__name__)


VERBOSE=True


if VERBOSE:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    verboseprint = lambda *a, **k: None 



def binary_mask_to_polygon(binary_mask, tolerance):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour





def nuclei_info(n, image_segmentation, cat_id, image_id):
    if n!=0:
        binary_mask = np.where(image_segmentation!=n, 0, image_segmentation)
        binary_mask = np.where(image_segmentation==n, 1, binary_mask)
        #images calculations
        tolerance = 1.5
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        bounding_box = mask.toBbox(binary_mask_encoded)
        area = mask.area(binary_mask_encoded)
        
        annotation_info = {'id': int(n),
                   'category_id': int(cat_id) ,
                   'iscrowd': int(0),
                   'segmentation': segmentation,
                   'image_id': int(image_id),
                   'area': int(area.tolist()),
                   'bbox': bounding_box.tolist()
                   }

        if area.tolist()!=0:
            return annotation_info



    

def coco_output(ROOT_DIR = None, image_files=None, z_slice=None, output=None):
    img_DIR = os.path.join(ROOT_DIR, "img/")
    masks_DIR = os.path.join(ROOT_DIR, "masks/")
    
    
    INFO = {
        "info": "",
        "description": "",
        "url": " ",
        "version": "0.1.0",
        "year": 2020,
        "contributor": "TR",
        "date_created": "May 2020",
    }


    LICENSES = [    {
            "id": 1,
            "name": "",
            "url": " "
        }
    ]
        
        
    CATEGORIES = [
        {
            'supercategory': 'nucleus',
            'id': 1,
            'name': 'nucleus'
            
        },
    ]


    # create skeleton
    coco_out = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
        }



    image_id = 0
    segmentation_id = 1 # these do not restart for each image... unique identifier in all json

    #
    image_files = np.sort(image_files)
    verboseprint(image_files)
    # go through each image
    image_filename = image_files[0]
    verboseprint(image_filename)
    image = Image.open(img_DIR+image_filename)
    width, height = image.size

    image_info = {'id': image_id,
                  'file_name':image_filename,
                  'height': height,
                  'width': width,
                 'zplane':z_slice}

    coco_out["images"].append(image_info)


    #for root, _, files in os.walk(ANNOTATION_DIR):
    #image_segmentation = glob.glob(masks_DIR+'/*'+str.split(image_filename,".")[0]+'_*.png')
    #image_segmentation = glob.glob(masks_DIR+'/*'+image_filename)
    image_segmentation = masks_DIR+"z"+str(z_slice)+"z_"+image_filename
    image_segmentation = np.asarray(Image.open(image_segmentation)).astype(np.int32)
    nuclei = np.unique(image_segmentation)
    verboseprint(len(nuclei))
    cat_id = 1
    cat_name = 'nucleus' # this could have here another loop for more cats 
    
    
    return coco_out,nuclei,image_segmentation,cat_id,image_id


  

def main(ROOT_DIR = None, image_files=None, z_slice=None, output=None):
    
    print(ROOT_DIR)
    print(z_slice)
    print(output)
    
    
    if (type(image_files) is list):
        pass
    elif image_files is None:
        print("Assuming image files from img dir.")
        image_files = os.listdir(img_DIR)
    else:
        print("Error!")

    if ROOT_DIR is None:
        print("Supply root directory. Inside should be the img/ and masks/ folders.")
    if output is None:
        print("Supply output json path and filename.")
    
    if z_slice is None:
        print("Supply z-slice of original image. zero if none.")

    
    coco_out, nuclei, image_segmentation, cat_id, image_id = coco_output(ROOT_DIR = ROOT_DIR, image_files=image_files, z_slice=z_slice, output=output)
    
    #print(coco_out)
    
    tic=timeit.default_timer()
    p = Pool(cpu_count(), maxtasksperchild=100)
    func = partial(nuclei_info,image_segmentation=image_segmentation, cat_id=cat_id,image_id=image_id)
    annotations = p.map(func, nuclei)
    p.close()
    p.join()
    toc=timeit.default_timer()
    print('The time for annotation info was {:.2f} sec'.format(toc - tic))
    
    [coco_out["annotations"].append(x) for x in annotations if x is not None]
    
    #print(annotations[0])
    #print(annotations[1])
    
    with open(output, 'w') as output_json_file:
        json.dump(coco_out, output_json_file,indent=4)
    
    print("Done saving JSON!")
    
    
if __name__ == '__main__':
     main()