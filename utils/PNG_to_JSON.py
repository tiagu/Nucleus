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
from tqdm import tqdm


print(__name__)


VERBOSE=False


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



def coco_output(ROOT_DIR = None, image_files=None, output=None):
    
    if (type(image_files) is list):
        pass
    elif image_files is None:
        print("Image files expects a list of image paths+filenames")
    else:
        print("Error!")

    if ROOT_DIR is None:
        print("Supply root directory. Inside should be the img/ and masks/ folders.")
    if output is None:
        print("Supply output json path and filename.")
    
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
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
        }



    image_id = 0
    segmentation_id = 1 # these do not restart for each image... unique identifier in all json

    image_files = os.listdir(img_DIR)
    image_files = np.sort(image_files)
    verboseprint(image_files)
    # go through each image
    for image_filename in image_files:
        verboseprint(image_filename)
        image = Image.open(img_DIR+image_filename)
        width, height = image.size
        
        image_info = {'id': image_id,
                      'file_name':image_filename,
                      'height': height,
                      'width': width}
                      
        coco_output["images"].append(image_info)


        #for root, _, files in os.walk(ANNOTATION_DIR):
        #image_segmentation = glob.glob(masks_DIR+'/*'+str.split(image_filename,".")[0]+'_*.png')
        #image_segmentation = glob.glob(masks_DIR+'/*'+image_filename)
        image_segmentation = masks_DIR+'/'+image_filename
        image_segmentation = np.asarray(Image.open(image_segmentation)).astype(np.int32)
        nuclei = np.unique(image_segmentation)
        verboseprint(len(nuclei))
        cat_id = 1
        cat_name = 'nucleus' # this could have here another loop for more cats 
        #[x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

        #within a cat_id count instances

        for n in tqdm(nuclei):
            #print(annotation_filename)
            #annotation_info = {}

            #plt.figure(figsize=(5,5))
            #plt.subplot(1,1,1)
            #I = io.imread(annotation_filename)
            #plt.axis('off')
            #plt.imshow(I)
            if n!=0:
                binary_mask = np.where(image_segmentation!=n, 0, image_segmentation)
                binary_mask = np.where(image_segmentation==n, 1, binary_mask)
                #images calculations
                tolerance = 1.5
                segmentation = binary_mask_to_polygon(binary_mask, tolerance)
                binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
                bounding_box = mask.toBbox(binary_mask_encoded)
                area = mask.area(binary_mask_encoded)

                annotation_info = {'id': segmentation_id,
                                   'category_id': cat_id ,
                                   'iscrowd': 0,
                                   'segmentation': segmentation,
                                   'image_id': image_id,
                                   'area': area.tolist(),
                                   'bbox': bounding_box.tolist()
                                   }

                if area.tolist()!=0:
                    coco_output["annotations"].append(annotation_info)
                    segmentation_id = segmentation_id + 1
        
        #print(annotation_info)
        image_id = image_id + 1
        


    with open(output, 'w') as output_json_file:
        json.dump(coco_output, output_json_file,indent=4)



    print("Done!")



if __name__ == '__main__':
     main()