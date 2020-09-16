python inference_surface_normal.py --checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/DFPN_SR.ckpt' \
                                   --sr_checkpoint_path '/mars/mnt/oitstorage/khiem_storage/ECCV_results/release/SR_only.ckpt' \
                                   --log_folder './demo_results' \
                                   --operation 'inference' \
                                   --batch_size 8 \
                                   --net_architecture 'sr_dfpn' \
                                   --test_dataset './demo_dataset'