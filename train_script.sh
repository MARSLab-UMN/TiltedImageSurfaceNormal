python train_test_generalized_surface_normal.py --checkpoint_path './checkpoints/DFPN_TAL.ckpt' \
                                                --log_folder './release_train/dfpn_rectified/' \
                                                --operation 'train_TAL_loss' \
                                                --learning_rate 1e-4 \
                                                --batch_size 8 \
                                                --train_dataset 'scannet_2dof_alignment' \
                                                --test_dataset 'scannet_2dof_alignment' \
                                                --net_architecture 'dfpn' \
                                                --augmentation 'warp_input'

python train_test_generalized_surface_normal.py  --rectified_checkpoint_path './release_train/dfpn_rectified/model-epoch-00000-iter-06000.ckpt' \
                                                 --log_folder './release_train/sr_dfpn/' \
                                                 --operation 'train_sr_fpn_full' \
                                                 --learning_rate 5e-5 \
                                                 --batch_size 8 \
                                                 --augmentation 'random_warp_input' \
                                                 --train_dataset 'scannet_2dof_alignment' \
                                                 --net_architecture 'sr_dfpn'