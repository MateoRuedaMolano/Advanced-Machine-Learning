# videosegmentation
# STESA:GRU Spatio-Temporal Self-Attended Gated Recurrent Units for Video Object Segmentation.

This code is based on RVOS and the modifications are done on their public code

## Installation 
-Clone the repo

```shell
git clone https://github.com/IBIO4615-2019/videosegmentation.git
```

-Install requirements ```pip install -r requirements.txt```

## Data

### YouTube-VOS 2018 version

Download the YouTube-VOS dataset from their website. Create a folder named ```databases```in the parent folder of the root directory and put there the database in a folder  named ```YouTubeVOS```. The root directory (```STESA-GRU```folder) and the ```databases``` folder should be in the same directory. 

### DAVIS 2017
Download the DAVIS 2017 dataset from their website at 480p resolution. Put there the database in the directory ```databases``` in  a folder named ```DAVIS2017```. The root directory (```STESA-GRU```folder) and the ```databases``` folder should be in the same directory. 

### LMDB data indexing

To highly speed the data loading we recommend to generate an LMDB indexing of it by doing:
```
python dataset_lmdb_generator.py -dataset=youtube
```
or
```
python dataset_lmdb_generator.py -dataset=davis2017
```
depending on the dataset you are using.

## Training

We provide a scripts folder with the .sh files to run the code

- Train the model for davis dataset with ```bash train_davis.sh``` or ```bash train_davis_auxiliary_loss.sh``` deppending on your preferences
- Train the model for youtube dataset with ```bash train_youtube.sh``` or ```bash train_youtube_auxiliary_loss.sh``` deppending on your preferences.
- Train the model for davis dataset with a pretrained model from youtube with ```bash train_davis_from_youtube.sh````or ```bash train_davis_from_youtube_aùxiliary_loss.sh``` deppending on the model that you pretrain

For all cases please change the model_name and de transfer_from when apply.

## Evaluation

We provide bash scripts to  evaluate models for the YouTube-VOS and DAVIS 2017 datasets. You can find them under the ```scripts``` folder.

Furthermore, in the ```src``` folder, ```prepare_results_submission.py```and ```prepare_results_submission_davis``` can be applied to change the format of the results into those accepted in the codalab server

## Demo

You can run demo.py to do generate the segmentation masks of a video. Just do:
```
python demo.py -model_name one-shot-model-davis --overlay_masks
```
and it will generate the resulting masks.

## Pretrained models

Download weights for models trained with:

- [YouTube-VOS]
- [DAVIS-2017]

Extract and place the obtained folder under ```models``` directory.
You can then run evaluation scripts with the downloaded model by setting ```args.model_name``` to the name of the folder.
