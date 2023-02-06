# Nucleus
Uses Detectron2 to detect individual nuclei in crowded immunofluorescence images.

## How to
To detect nuclei in your own image of interest just follow the notebook [Nucleus_Predict](https://github.com/tiagu/Nucleus/blob/master/notebooks/Nucleus_Predict.ipynb?flush_cache=true).

Please note, we mostly tested:
1) DAPI nuclear stains
2) Resolutions of 3.5 to 6 pixels/&mu;m (typically with 40-60X objectives on a Leica SP8/ Zeiss LSM710)
3) The size of the images should be a multiple of 128, e.g. 1024x2084.


Ideally, a 256x256 pixels crop of your images should look similar to the images the network was trained on. See below some examples.

<p align="center">
  <img width=512 alt="portfolio_view" src="https://github.com/tiagu/Nucleus/blob/master/outputs/misc/Nucleus-GIF.gif">
</p>


## Data used
This table shows the main datasets used to train and validate Detectron2. These are available at XXXXXXXXXX.

|   | #images  |  #instances | size  | comments  |
|---|---|---|---|---|
nucleus_train |	6 |	221 | 256*256 | in vitro hESC assay
nucleus_val (validation) | 4 | 141 | 256*256 | in vitro hESC assay
kromp_ 2019 | 52 | 1,704 | 640*512 | curated from Kromp et al. (2019)
segm_512 | 3 | 566 | 512*512 | in vitro hESC assay
SC_human | 4 | X | 256*256 | Spinal Cord sections
SC_mouse | 6 | X | 256*256 | Spinal Cord sections
SC_sections (validation) | 5 | X | 256*256 | Spinal Cord sections

<br/><br/>
Distribution of the maximum length of the nuclei in the different datasets.<p align="center">
<img width=420 src="https://github.com/tiagu/Nucleus/blob/master/outputs/misc/Nucleus_data_dimensions.png">
</p>


## Options

At present access to GPU and pytorch are still required. 

A) Stitching method

- [x] no stitiching
- [x] stitching v1 (overkill and slow)
- [x] stitching v2 (hopefully better than v1)

B) Overlayed image with segmentation (slow) or just output masks (faster)

C) 3D consolidate nuclei by joinning masks across z

[further options: XXXXXXXXXX]


### References



