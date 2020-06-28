# Nucleus
Uses Detectron2 to detect single nuclei in crowded immunofluorescence images.

## How to
To detect nuclei in your own image of interest just follow the notebook [Nucleus_Predict](https://github.com/tiagu/Nucleus/blob/master/Nucleus_Predict.ipynb?flush_cache=true).

Please note, we mostly tested:
1) DAPI nuclear stains
2) Resolutions of 3.5 to 6 pixels/&mu;m (typically with 40-60X objectives on a Leica SP8/ Zeiss LSM710)
3) The size of the images should be a multiple of 256, e.g. 1024x2084.


Ideally, a 256x256 pixels square cropped from your images should look similar to the images the network was trained on. Below are some examples.

<p align="center">
  <img width="512" height="256" src="https://github.com/tiagu/Nucleus/blob/master/examples/Nucleus-GIF.gif">
</p>


## Data used
This table shows the main datasets used to train and validate Detectron2. These are available at XXXXXXXXXX.

|   | #images  |  #instances | size  | comments  |
|---|---|---|---|---|
nucleus_train |	6 |	221 | 256*256 | in vitro hESC assay
nucleus_val (validation) | 4 | 141 | 256*256 | in vitro hESC assay
kromp_ 2019 | 52 | 1,704 | 640*512 | curated from Kromp et al. (2019)
segm_512 | 3 | 566 | 512*512 | in vitro hESC assay
SC_sections (human) | 4 | X | 256*256 | Spinal Cord sections
SC_sections (mouse) | 6 | X | 256*256 | Spinal Cord sections
SC_sections (validation) | 5 | X | 256*256 | Spinal Cord sections

<br/><br/>
Distribution of the maximum length of the nuclei in the different datasets.<p align="center">
<img width="495" height="350" src="https://github.com/tiagu/Nucleus/blob/master/examples/Nucleus_data_dimensions.png">
</p>


## Details and requirements

Access to GPU. 

A) In case of images larger than 256 pixels:

- [x] no stitiching
- [x] stitching v1 (overkill and slow)
- [x] stitching v2 (hopefully better than v1)
- [ ] stitching touching polygons (faster but perhaps less accurate?)

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

