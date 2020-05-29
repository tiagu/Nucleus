# Nucleus
Uses Detectron2 to detect single nuclei in crowded immunofluorescence images.

## Data used
This table shows the main datasets used to train and validate Detectron. These are available at XXXXXXXXXX.

|   | #images  |  #instances | size  | comments  |
|---|---|---|---|---|
nucleus_train |	6 |	221 | 256*256 | in vitro hESC assay
nucleus_val | 4 | 141 | 256*256 | in vitro hESC assay
kromp_ 2019 | 52 | 1,704 | 640*512 | curated from Kromp et al. (2019)
segm_512 | 3 | 566 | 512*512 | in vitro hESC assay
SC_sections | X | X | 256*256 | Spinal Cord sections
|   |   |   |   |   |

## How to
To detect nuclei in your own image of interest just follow the notebook [Nucleus_Predict](https://github.com/tiagu/Nucleus/blob/master/Nucleus_Predict.ipynb?flush_cache=true).


Please note, we mostly tested:
1) DAPI nuclear stains
2) 40-60X objectives (Leica SP8/ Zeiss 710)
3) Size of image should be a multiple of 256 (256, 512, 1024, 2084, ...)


If you crop your image to a 256x256 pixels square, these some typical images the network is used to predict:


3x3 image grid


## Details 

Access to GPU. 

Options:

A) In case of images larger than 256 pixels:

- no stitiching
- stitching v1 (overkill and slow)
- stitching v2 (hopefully better than v1)
- stitching touching polygons (faster but perhaps less accurate?)

B) Overlayed image with segmentation or just output masks (faster)


C) If you dare to attempt a tif 3D stack as input. Additionaly joinning nuclei across z.

--3D=True/False

[further options: XXXXXXXXXX]


## To do

- stitching v2
- stitching touching polygons [this does not generate additional images at the middle of boundaries. Just scans edges and merges nuclei if their areas intersect sufficiently with a specificed margin.]
- 3D stacks


### References

XXXXX

