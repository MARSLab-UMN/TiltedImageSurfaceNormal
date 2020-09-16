#python train_test_generalized_surface_normal.py --checkpoint_path '/mars/mnt/dgx/ECCV2020_results/plain_fpn_dorn_robust_acos_bs32_standard/model-final.ckpt' \
#                                               --log_folder '/mars/mnt/oitstorage/khiem_storage/ECCV_results/plainfpn_rectified_1st_trial/' \
#                                               --operation 'train_robust_acos_loss' \
#                                               --learning_rate 1e-4 \
#                                               --batch_size 8 \
#                                               --train_dataset 'scannet_2dof_alignment' \
#                                               --net_architecture 'plain_fpn' \
#                                               --augmentation 'warp_input'

#python train_test_generalized_surface_normal.py   --rectified_checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/plainfpn_rectified_1st_trial/model-epoch-00017-iter-24000.ckpt' \
#                                                  --log_folder '/mars/mnt/oitstorage/khiem_storage/ECCV_results/sr_pfpn_full_1st_trial/' \
#                                                  --operation 'train_sr_fpn_full' \
#                                                  --learning_rate 5e-5 \
#                                                  --batch_size 8 \
#                                                  --augmentation 'random_warp_input' \
#                                                  --train_dataset 'scannet_2dof_alignment' \
#                                                  --net_architecture 'sr_pfpn'

#python train_test_generalized_surface_normal.py   --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/sr_pfpn_full_1st_trial/model-epoch-00002-iter-06000.ckpt' \
#                                                  --log_folder '/mars/mnt/oitstorage/khiem_storage/ECCV_results/sr_pfpn_full_3rd_trial/' \
#                                                  --operation 'train_sr_fpn_full' \
#                                                  --learning_rate 5e-5 \
#                                                  --batch_size 8 \
#                                                  --augmentation 'random_warp_input' \
#                                                  --train_dataset 'scannet_2dof_alignment' \
#                                                  --net_architecture 'sr_pfpn'

python train_test_generalized_surface_normal.py   --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/sr_pfpn_full_3rd_trial/model-epoch-00003-iter-00000.ckpt' \
                                                  --log_folder '/mars/mnt/oitstorage/khiem_storage/ECCV_results/sr_pfpn_full_4th_trial/' \
                                                  --operation 'train_sr_fpn_full' \
                                                  --learning_rate 5e-5 \
                                                  --batch_size 8 \
                                                  --augmentation 'random_warp_input' \
                                                  --train_dataset 'scannet_2dof_alignment' \
                                                  --net_architecture 'sr_pfpn'