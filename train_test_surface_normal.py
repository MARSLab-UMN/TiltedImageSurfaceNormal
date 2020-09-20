import torch
import numpy as np
import argparse
import os


from data import create_dataset_loader, data_augmentation
from losses import compute_surface_normal_angle_error
from model import create_network, forward_cnn
from utils import log, log_normal_stats, check_nan_ckpt
from warping_2dof_alignment import Warping2DOFAlignment


def parsing_configurations():
    parser = argparse.ArgumentParser(description='Train/Test surface normal estimation')
    parser.add_argument('--log_folder', type=str, default='')
    parser.add_argument('--operation', type=str, default='evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--rectified_checkpoint_path', type=str, default='')
    parser.add_argument('--sr_checkpoint_path', type=str,
                        default='/mars/mnt/oitstorage/tien_storage/FPN_warping/spatial_rectifier_2nd_trial/model-epoch-00016-iter-24000.ckpt')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_dataset', type=str, default='./data/scannet_standard_train_test_val_split.pkl')
    parser.add_argument('--test_dataset', type=str, default='./data/scannet_standard_train_test_val_split.pkl')
    parser.add_argument('--net_architecture', type=str, default='dorn')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--augmentation', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--print_every_x_iterations', type=int, default=60)
    parser.add_argument('--evaluate_every_x_iterations', type=int, default=600)
    parser.add_argument('--save_ckpt_every_x_iterations', type=int, default=6000)

    args = parser.parse_args()

    config = {'ARCHITECTURE': args.net_architecture,
              'AUGMENTATION': args.augmentation,
              'BATCH_SIZE': args.batch_size,
              'CKPT_PATH': args.checkpoint_path,
              'EVAL_ITER': args.evaluate_every_x_iterations,
              'LEARNING_RATE': args.learning_rate,
              'LOG_FOLDER': args.log_folder,
              'MAX_EPOCH': args.max_epoch,
              'PRINT_ITER': args.print_every_x_iterations,
              'OPERATION': args.operation,
              'OPTIMIZER': args.optimizer,
              'RECTIFIED_CKPT_PATH': args.rectified_checkpoint_path,
              'SAVE_ITER': args.save_ckpt_every_x_iterations,
              'SR_CKPT_PATH': args.sr_checkpoint_path,
              'TRAIN_DATASET': args.train_dataset,
              'TEST_DATASET': args.test_dataset}
    return config


total_normal_errors = None


def accumulate_prediction_error(sample_batched, angle_error_prediction):
    global total_normal_errors
    mask = sample_batched['mask'] > 0
    if total_normal_errors is None:
        total_normal_errors = angle_error_prediction[mask].data.cpu().numpy()
    else:
        total_normal_errors = np.concatenate((total_normal_errors, angle_error_prediction[mask].data.cpu().numpy()))


if __name__ == '__main__':
    # Step 1. Configuration file
    config = parsing_configurations()

    # Create logger file
    training_loss_file = None
    evaluate_stat_file = None
    if config['LOG_FOLDER'] != '':
        if not os.path.exists(config['LOG_FOLDER']):
            os.makedirs(config['LOG_FOLDER'])
        training_loss_file = open(config['LOG_FOLDER'] + '/training_loss.txt', 'w')
        evaluate_stat_file = open(config['LOG_FOLDER'] + '/evaluate_stat.txt', 'w')
    log(config, training_loss_file)
    log(config, evaluate_stat_file)

    # Step 2. Create dataset loader
    train_dataloader, test_dataloader, val_dataloader = create_dataset_loader(config)

    # Step 3. Create cnn
    cnn = create_network(config)

    if config['CKPT_PATH'] is not '':
        print('Loading checkpoint from %s' % config['CKPT_PATH'])
        cnn.load_state_dict(torch.load(config['CKPT_PATH']))

    # Step 4. Create optimizer
    optimizer = None
    if 'train' in config['OPERATION']:
        if config['OPTIMIZER'] == 'adam':
            optimizer = torch.optim.Adam(cnn.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.999))
        else:
            raise Exception('Optimizer not implemented!')

    # Step 5. Create warper input:
    warper = Warping2DOFAlignment()

    # Step 6. Learning loop
    if 'train' in config['OPERATION']:
        # Store a pointer to the last checkpoint
        last_ckpt = {'epoch': None, 'iter': None}

        for epoch in range(0, CONFIG['MAX_EPOCH']):
            for iter, sample_batched in enumerate(train_dataloader):
                cnn.train()

                sample_batched = {data_key: sample_batched[data_key].cuda() for data_key in sample_batched}

                if config['AUGMENTATION'] != '':
                    sample_batched = data_augmentation(sample_batched, config, warper, epoch, iter)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Step 6b: Forward pass
                output_prediction = forward_cnn(sample_batched, cnn, config)

                # Step 6c: Compute loss
                losses, logging_losses = compute_surface_normal_angle_error(sample_batched,
                                                                            output_prediction,
                                                                            mode=config['OPERATION'],
                                                                            angle_type='delta')
                # Step 6d: Backward pass and update
                losses.backward()
                optimizer.step()

                # Step 6e. Print loss value
                if iter % CONFIG['PRINT_ITER'] == 0:
                    log('Epoch %d, Iter %d, Loss %.4f' % (epoch, iter, logging_losses), training_loss_file)

                # Step 6f. Print robust evaluation stats
                if iter % CONFIG['EVAL_ITER'] == 0:
                    # Reload closest checkpoint if hit nan
                    if check_nan_ckpt(cnn):
                        cnn.load_state_dict(torch.load(config['LOG_FOLDER'] + '/model-epoch-%05d-iter-%05d.ckpt'
                                                       % (last_ckpt['epoch'], last_ckpt['iter'])))
                        optimizer.load_state_dict(
                            torch.load(config['LOG_FOLDER'] + '/optimizer-epoch-%05d-iter-%05d.ckpt'
                                       % (last_ckpt['epoch'], last_ckpt['iter'])))
                        log('Getting Nan, reloading model from ep: %d, iter: %d'
                            % (last_ckpt['epoch'], last_ckpt['iter']), training_loss_file)

                    evaluation_mode = 'evaluate' + config['OPERATION'][len('train'):] if 'mix_loss' in config['OPERATION'] else 'evaluate'
                    total_normal_errors = None

                    with torch.no_grad():
                        print('<EVALUATION MODE:', evaluation_mode, '>')
                        cnn.eval()
                        for _, eval_batch in enumerate(test_dataloader): # TODO: val_dataloader
                            eval_batch = {data_key: eval_batch[data_key].cuda() for data_key in eval_batch}
                            if config['AUGMENTATION'] == 'warp_input':
                                eval_batch = data_augmentation(eval_batch, config, warper, epoch, iter)
                            output_prediction = forward_cnn(eval_batch, cnn, config)
                            if 'sr' in config['ARCHITECTURE']:
                                surfacenormal_pred = output_prediction['n']
                            else:
                                surfacenormal_pred = output_prediction
                            angle_error_prediction = compute_surface_normal_angle_error(eval_batch,
                                                                                        surfacenormal_pred,
                                                                                        mode=evaluation_mode,
                                                                                        angle_type='delta')
                            accumulate_prediction_error(eval_batch, angle_error_prediction)
                        log_normal_stats(epoch, iter, total_normal_errors, evaluate_stat_file)

                # Step 6g. Save checkpoints into file
                if iter % CONFIG['SAVE_ITER'] == 0:
                    path = config['LOG_FOLDER'] + '/model-epoch-%05d-iter-%05d.ckpt' % (epoch, iter)
                    torch.save(cnn.state_dict(), path)
                    path = config['LOG_FOLDER'] + '/optimizer-epoch-%05d-iter-%05d.ckpt' % (epoch, iter)
                    torch.save(optimizer.state_dict(), path)
                    last_ckpt['epoch'] = epoch
                    last_ckpt['iter'] = iter
    else:
        cnn.eval()
        total_normal_errors = None
        with torch.no_grad():
            for iter, sample_batched in enumerate(test_dataloader):
                sample_batched = {data_key:sample_batched[data_key].cuda() for data_key in sample_batched}
                output_prediction = forward_cnn(sample_batched, cnn, config)
                angle_error_prediction = compute_surface_normal_angle_error(sample_batched, output_prediction,
                                                                            mode=config['OPERATION'], angle_type='delta')
                accumulate_prediction_error(sample_batched, angle_error_prediction)

        # TOTAL error
        print('NORMAL ERROR STATS: ')
        log_normal_stats(0, 0, total_normal_errors)
