python ../src/train_previous_mask_auxiliary_loss.py -model_name=exp -RNN=GRU -NLB=True -dataset=davis2017 -batch_size=4 -length_clip=5 -base_model=resnet101 -max_epoch=40 --augment --resize -gpu_id=0
