# Nucleus
Pipeline to detect individual nuclei in crowded immunofluorescence images. Effective in 2D embryo sections and single-z plane images. Additionally, a range of tools are available for estimating 3D nuclei in whole-embryo, organoid, or embryo model confocal microscopy stacks.

<p align="center">
  <img width=512 alt="portfolio_view" src="https://github.com/tiagu/Nucleus/blob/master/utils/misc/3d_micropattern.png">
</p>

If you use this repository in your research, please cite:

Rito, T. *et al.* Timely TGFβ signalling inhibition induces notochord. Nature 637, 673–682 (2025). DOI: [10.1038/s41586-024-08332-w](https://doi.org/10.1038/s41586-024-08332-w).

## How to
Please find installation notes below. To detect nuclei in your own image of interest I recommend you first following one of the two example notebooks. 

2D notebook example: [Nucleus_Predict_2D](https://github.com/tiagu/Nucleus/blob/master/notebooks/Nucleus_predict_2D.ipynb?flush_cache=true). Example images [here](https://github.com/tiagu/Nucleus/blob/master/test/2D/masks/zmicropattern_zplane.tif_coco_out.png?flush_cache=true) and [here](https://github.com/tiagu/Nucleus/blob/master/test/2D/masks/zmouse_section.tif_coco_out.png?flush_cache=true).

3D notebook example: [Nucleus_Predict_3D](https://github.com/tiagu/Nucleus/blob/master/notebooks/Nucleus_predict_3D.ipynb?flush_cache=true).


Please note, we mostly tested:
1) DAPI nuclear stains
2) Resolutions of 3.5 to 6 pixels/&mu;m (typically with 40-60X objectives on a Leica SP8/ Zeiss LSM710)
3) The size of the images should be a multiple of 128, e.g. 1024x2084. There is a check in the pipeline for this.


Ideally, a 256x256 pixels crop of your images should look similar to the images the network was trained on. See below some examples.

<p align="center">
  <img width=512 alt="portfolio_view" src="https://github.com/tiagu/Nucleus/blob/master/utils/misc/Nucleus-GIF.gif">
</p>


## Data used
This table shows the main datasets used to train and validate our models. These are available [here](https://github.com/tiagu/Nucleus/blob/master/utils/misc/Nucleus_data.gz?flush_cache=true).

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
<img width=420 src="https://github.com/tiagu/Nucleus/blob/master/utils/misc/Nucleus_data_dimensions.png">
</p>


## Instalation

Make sure you have access to GPU. This package uses python 3.9.

Download PyTorch models for Nucleus at
https://zenodo.org/records/15595710


``` bash
git clone https://github.com/tiagu/Nucleus

uv venv nucleus-2 --seed --python python3.9

source nucleus-2/bin/activate

uv pip install torch==1.10 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

uv pip install --upgrade pip

# deal with new incompatible versions
uv pip install Pillow==9.5 jupyterlab
uv pip install numpy==1.21.6 contourpy matplotlib scikit-image scipy pandas opencv-python tqdm

python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

```

For slurm users, please see [run_nucleus.sh](https://github.com/tiagu/Nucleus/blob/master/utils/run_nucleus.sh?flush_cache=true) on utils folder.


### References

[1] https://scikit-image.org/

[2]https://github.com/facebookresearch/detectron2

[3]https://opencv.org/

[4]https://pytorch.org/

