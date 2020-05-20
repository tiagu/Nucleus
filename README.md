# Nucleus
Uses Detectron2 to detect single nuclei in crowded immunofluorescence images.

## Data used
This table shows the main datasets used to train and validate Detectron. These are available at XXXXXXXXXX.





## How to
To detect nuclei in your own image of interest just follow the notebook XXXXXXXXXX. 

Please note, we mostly tested:
1) DAPI nuclear stains
2) 40-60X objectives ()
3) Size of image should be a multiple of 256 (256, 512, 1024, 2084, ...)


If you crop your image to a 256x256 pixels square, these some typical images the network is used to predict:


3x3 image grid
## Details 

Options:

A) In case of images larges than 256 pixels:
- no stitiching
- stitching v1 (overkill and slow)
- stitching v2 (hopefully better than v1)
- stitching touching polygons (faster but perhaps less accurate)

B) Overlayed image with segmentation or just output masks (faster)

C) Giving it tif 3D stack. Additionaly joinning nuclei across z.

[further options: XXXXXXXXXX]


## To do

- stitching v2
- stitching touching polygons
- 3D stacks


### References

XXXXX

