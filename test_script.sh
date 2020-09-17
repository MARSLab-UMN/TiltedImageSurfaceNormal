# DFPN+TAL
## Tilted Images on Tilt-RGBD
python train_test_generalized_surface_normal.py \
                     --checkpoint_path './checkpoints/DFPN_TAL.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_biased_viewing_directions' \
                     --net_architecture 'dfpn' \
                     --batch_size 128

## Gravity-aligned Images on Tilt-RGBD
python train_test_generalized_surface_normal.py \
                     --checkpoint_path './checkpoints/DFPN_TAL.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
                     --net_architecture 'dfpn' \
                     --batch_size 128

## NYUv2
python train_test_generalized_surface_normal.py \
                     --checkpoint_path './checkpoints/DFPN_TAL.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'nyud' \
                     --net_architecture 'dfpn' \
                     --batch_size 128

## ScanNet
python train_test_generalized_surface_normal.py \
                     --checkpoint_path './checkpoints/DFPN_TAL.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'scannet_standard' \
                     --net_architecture 'dfpn' \
                     --batch_size 128

# ====================================================================================================================

# DFPN+TAL+SR
## Tilted Images on Tilt-RGBD
python train_test_generalized_surface_normal.py \
                     --checkpoint_path './checkpoints/DFPN_TAL_SR.ckpt' \
                     --sr_checkpoint_path './checkpoints/SR_only.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_biased_viewing_directions' \
                     --net_architecture 'sr_dfpn' \
                     --batch_size 128

## Gravity-aligned Images on Tilt-RGBD
python train_test_generalized_surface_normal.py \
                     --checkpoint_path './checkpoints/DFPN_TAL_SR.ckpt' \
                     --sr_checkpoint_path './checkpoints/SR_only.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
                     --net_architecture 'sr_dfpn' \
                     --batch_size 128
