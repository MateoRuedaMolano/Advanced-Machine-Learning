python ../src/train_previous_mask_auxiliary_loss_youtube.py -model_name=exp -RNN=GRU -NLB=True -dataset=youtube -batch_size=4 -length_clip=5 -base_model=resnet101 -max_epoch=20 --augment --resize -gpu_id=0