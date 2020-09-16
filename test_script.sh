## DFPN+TAL+SR
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN_SR.ckpt' \
#                     --sr_checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/SR_only.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_biased_viewing_directions' \
#                     --net_architecture 'sr_dfpn' \
#                     --batch_size 128
#
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN_SR.ckpt' \
#                     --sr_checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/SR_only.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
#                     --net_architecture 'sr_dfpn' \
#                     --batch_size 128

# DFPN+TAL
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_biased_viewing_directions' \
#                     --net_architecture 'dfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
#                     --net_architecture 'dfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'nyud' \
#                     --net_architecture 'dfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'scannet_standard' \
#                     --net_architecture 'dfpn' \
#                     --batch_size 128

#=======================================================================================================================
# MFPN+TAL+SR
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/tien_storage/FPN_warping/sr_fpn_full_6th_trial/model-epoch-00006-iter-18000.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_biased_viewing_directions' \
#                     --net_architecture 'sr_mfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/oitstorage/tien_storage/FPN_warping/sr_fpn_full_6th_trial/model-epoch-00006-iter-18000.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
#                     --net_architecture 'sr_mfpn' \
#                     --batch_size 128

## MFPN+TAL (overfit!)
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/dgx/FPN_results_new_pipeline/new_pipeline_acos_loss_2nd_trial/model-epoch-00019-iter-24000.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_biased_viewing_directions' \
#                     --net_architecture 'mfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/dgx/FPN_results_new_pipeline/new_pipeline_acos_loss_2nd_trial/model-epoch-00019-iter-24000.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
#                     --net_architecture 'mfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/dgx/FPN_results_new_pipeline/new_pipeline_acos_loss_2nd_trial/model-epoch-00019-iter-24000.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'nyud' \
#                     --net_architecture 'mfpn' \
#                     --batch_size 128
#python train_test_generalized_surface_normal.py \
#                     --checkpoint_path '/mars/mnt/dgx/FPN_results_new_pipeline/new_pipeline_acos_loss_2nd_trial/model-epoch-00019-iter-24000.ckpt' \
#                     --operation 'evaluate' \
#                     --test_dataset 'scannet_standard' \
#                     --net_architecture 'mfpn' \
#                     --batch_size 128

#=======================================================================================================================
# PFPN+TAL+SR
python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/PFPN_SR.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_biased_viewing_directions' \
                     --net_architecture 'sr_pfpn' \
                     --batch_size 128
python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/PFPN_SR.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
                     --net_architecture 'sr_pfpn' \
                     --batch_size 128
# PFPN+TAL
python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/PFPN.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_biased_viewing_directions' \
                     --net_architecture 'pfpn' \
                     --batch_size 128
python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/PFPN.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
                     --net_architecture 'pfpn' \
                     --batch_size 128
python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/PFPN.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'nyud' \
                     --net_architecture 'pfpn' \
                     --batch_size 128
python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/PFPN.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'scannet_standard' \
                     --net_architecture 'pfpn' \
                     --batch_size 128