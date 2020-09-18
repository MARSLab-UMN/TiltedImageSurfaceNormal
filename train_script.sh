# Train DFPN + TAL
python train_test_generalized_surface_normal.py --log_folder './log/release_train/dfpn_tal/' \
                                                --operation 'train_TAL_loss' \
                                                --learning_rate 1e-4 \
                                                --batch_size 8 \
                                                --net_architecture 'dfpn' \
                                                --train_dataset 'scannet_standard' \
                                                --test_dataset 'scannet_standard'