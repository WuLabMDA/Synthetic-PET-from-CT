# Synthetic PET from CT improves diagnosis and prognosis for lung cancer: proof of concept

This is the code for the paper "Synthetic PET from CT improves diagnosis and prognosis for lung cancer: proof of concept" (Cell Report Medicine 2024)
<div align=center><img src="Figure/Figure1.png" width = "85%"/></div>


## Training

After having the CT and PET data arrays (512*512*7) in the "/data_7CHL/pix2pix_7Ch7/trainA and trainB":

> python train.py --name 'lr0002_L8000_PET_2p5_p8_maxx10_gamma_sep17data' --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Sep17' --lr 0.0002 --lambda_L1 8000 --batch_size 6 --n_epochs 40 

Note: If necessary, specify the GPU to use by setting CUDA_VISIBLE_DEVICES=0, for example.

## Test

To test the trained model on a folder containing nifti CT files, run:

> CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Folder_with_lung_CT_Nifti_files' --name 'checkpoints' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Result_folder'

Note: If necessary, specify the GPU to use by setting CUDA_VISIBLE_DEVICES=0, for example.

We uploaded the trained model which achieves the performance reported in the paper to the 'checkpoints' folder for your reference.

## Dataset

MDA-TRAIN/TEST/SCREENING : N/A 

TCIA-STANFORD : https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics 

LIDC-IDRI : https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254 

NSCLC-RT : https://www.cancerimagingarchive.net/ 



## Citation

If you find CC useful in your research, please consider citing:
```
@inproceedings{****,
  title={Synthetic PET from CT improves diagnosis and prognosis for lung cancer: proof of concept},
  author={****},
  booktitle={****},
  volume={****},
  number={****},
  pages={****},
  year={****}
}
```

## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/phillipi/pix2pix/tree/master). The generator architecture was borrowed from [ResUNetPlusPlus](https://github.com/DebeshJha/ResUNetPlusPlus).


