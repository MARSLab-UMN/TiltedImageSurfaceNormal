# Step 1. Train the Spatial Rectifier only
python train_test_surface_normal.py   --log_folder './results/SR_only/' \
                                      --operation 'train_SR_only' \
                                      --learning_rate 1e-4 \
                                      --batch_size 8 \
                                      --train_dataset './data/rectified_2dofa_framenet.pkl' \
                                      --augmentation 'random_warp_input' \
                                      --net_architecture 'spatial_rectifier'

# Step 2. Train Rectified Surface Normal Estimation network
python train_test_surface_normal.py   --log_folder './results/PFPN_rectified/' \
                                      --checkpoint_path './checkpoints/PFPN_TAL.ckpt' \
                                      --operation 'train_TAL_loss' \
                                      --learning_rate 1e-4 \
                                      --batch_size 8 \
                                      --train_dataset './data/rectified_2dofa_framenet.pkl' \
                                      --net_architecture 'pfpn' \
                                      --augmentation 'warp_input'

# Step 3. Train the full pipeline
python train_test_surface_normal.py   --log_folder './results/PFPN_SR_full/' \
                                      --sr_checkpoint_path './results/SR_only/model-latest.ckpt' \
                                      --rectified_checkpoint_path './results/PFPN_rectified/model-best.ckpt' \
                                      --operation 'train_sr_fpn_full' \
                                      --learning_rate 5e-5 \
                                      --batch_size 8 \
                                      --augmentation 'random_warp_input' \
                                      --train_dataset './data/full_2dofa_framenet.pkl' \
                                      --net_architecture 'sr_pfpn'