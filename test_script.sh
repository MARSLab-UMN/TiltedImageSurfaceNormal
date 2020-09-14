python train_test_generalized_surface_normal.py \
                     --checkpoint_path '/mars/mnt/oitstorage/tien_storage/FPN_warping/sr_fpn_full_5th_trial/model-epoch-00005-iter-24000.ckpt' \
                     --operation 'evaluate' \
                     --test_dataset 'kinect_azure_unseen_viewing_directions' \
                     --net_architecture 'sr_fpn' \
                     --batch_size 128