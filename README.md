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

# Dataset

MDA-TRAIN/TEST/SCREENING : N/A 
TCIA-STANFORD : https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics 
LIDC-IDRI : https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254 
NSCLC-RT : https://www.cancerimagingarchive.net/ 



# Citation

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





```bash 
docker build -t ctpet .
```

## Tensorboard

```bash 
tensorboard --logdir=runs --port=6004
```

## Training 
```bash 
CUDA_VISIBLE_DEVICES=7 python train.py --name 'lr0002_L500_PET_2p5_p8_maxx10' --lr 0.0002 --lambda_L1 500 --batch_size 6 --n_epochs 100 --preprocess_gamma 0
CUDA_VISIBLE_DEVICES=1 python testNifty.py --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Sep17/testA' --name 'lr0002_L500_PET_2p5_p8_maxx10' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 0 --results_dir '/home/msalehjahromi/pix2pix/checkpoints/lr0002_L500_PET_2p5_p8_maxx10/Test21'

CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Slide3_Sep18/testA' --name 'lr0002_L500_PET_2p5_p8_maxx10_slide3_gamma' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/home/msalehjahromi/pix2pix/checkpoints/lr0002_L500_PET_2p5_p8_maxx10_slide3_gamma/'

CUDA_VISIBLE_DEVICES=1 python train.py --name 'lr0002_L8000_PET_2p5_p8_maxx10_gamma_sep17data' --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Sep17' --lr 0.0002 --lambda_L1 8000 --batch_size 6 --n_epochs 40 --preprocess_gamma 1

CUDA_VISIBLE_DEVICES=0 python train.py --name 'lr0002_L6000_SUV_max7_slide3_Oct29' --lr 0.0002 --lambda_L1 6000 --batch_size 5 --n_epochs 100 --preprocess_gamma 1

CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29/testA'

```

## other Testing 
```bash 

CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/Folder_containing_Nifty_files' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/Destination_folder'


CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/TCIA/Only_CTNEW' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/TCIA/PET_predTCIA'
```

## Testing on NIFTI files (3D)
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot '/Data/CTtoPET/pix2pix_5_1' --gpu_ids '0' CTPET_5ChlWithSld2_ValTrainTest_lungOnly_May5/CT_Tr' --name experiment_name --model 'pix2pix'


## License

## Docker (personal)
```bash 

docker run -it --rm --gpus all --shm-size=150G --user $(id -u):$(id -g) --cpuset-cpus=175-199 \
-v /rsrch1/ip/msalehjahromi/codes/CTtoPET/pix2pix:/home/msalehjahromi/pix2pix \
--name pix2pix1 pix2pix:latest


  GPU           CPUs

   0           0-24

   1           25-49

   2           50-74

   3           75-99

   4           100-124

   5           125-149

   6           150-174

   7           175-199




CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/LIDC/2_LIDC_Nifty_modified' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/home/msalehjahromi/pix2pix/checkpoints/lr0004_L8000_SUV_max7_slide3_Oct29_noNoise/2_LIDC_Nifty_modified'


CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/TCIA/Only_CTNEW' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/TCIA/PET_predTCIA'




For Eman:
docker run -it --rm --gpus all --shm-size=150G  --user $(id -u):$(id -g) --cpuset-cpus=0-24 \
-v /rsrch1/ip/msalehjahromi/codes/CTtoPET/pix2pix:/home/msalehjahromi/pix2pix \
-v /rsrch1/ip/msalehjahromi/data/:/Data \
--name pix2pix1 pix2pix:latest

CUDA_VISIBLE_DEVICES=0 python testNifty.py --dataroot '/Data/Eman' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/EmanP/'

#for non_smoker:
docker run -it --rm --gpus all --shm-size=150G  --user $(id -u):$(id -g) --cpuset-cpus=200-224 \
-v /rsrch1/ip/msalehjahromi/codes/CTtoPET/pix2pix:/home/msalehjahromi/pix2pix \
-v /rsrch1/ip/msalehjahromi/papers/CT2PET/Revision/data_ns/:/Data \
--name pix2pix1 pix2pix:latest

CUDA_VISIBLE_DEVICES=6 python testNifty.py --dataroot '/Data/Sybil_big3' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/Sybil_PETPred/'
CUDA_VISIBLE_DEVICES=6 python testNifty.py --dataroot '/Data/Sybil_lessp7' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoise' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/Sybil_PETPred/'




CUDA_VISIBLE_DEVICES=6 python train.py --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoiseJuly' --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29' --lr 0.0002 --lambda_L1 8000 --batch_size 6 --n_epochs 40 --preprocess_gamma 1







## Running Only on ICON, bad one
docker run -it --rm --gpus all --shm-size=150G --user $(id -u):$(id -g) --cpuset-cpus=175-199 \
-v /rsrch1/ip/msalehjahromi/codes/CTtoPET/pix2pix:/home/msalehjahromi/pix2pix \
-v /rsrch1/ip/msalehjahromi/data/:/Data \
-v /rsrch7/wulab/Mori/ICON_PROSPECT_TCIA/PROSPECT/test/:/Data1 \
--name pix2pix1 pix2pix:latest

CUDA_VISIBLE_DEVICES=6 python testNifty.py --dataroot '/Data1' --name 'lr0004_L8000_SUV_max7_slide3_Oct29_noNoiseJuly' --model 'pix2pix' --gpu_ids '0'  --mode 'test' --preprocess_gamma 1 --results_dir '/Data/Pix2pix_pred_noNoiseJuly/'
