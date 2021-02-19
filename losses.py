import torch
import numpy as np


def compute_surface_normal_angle_error(sample_batched, output_pred, mode='evaluate', angle_type='delta'):
    if 'Z' in sample_batched:
        surface_normal_pred = output_pred
        if mode == 'evaluate':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'])
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            return torch.acos(prediction_error) * 180.0 / np.pi

        elif mode == 'train_L2_loss':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            mask = sample_batched['mask'] > 0
            mask = mask.detach()
            return -torch.sum(prediction_error[mask]), 1.0-torch.mean(prediction_error[mask])

        elif mode == 'train_AL_loss':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            acos_mask = mask.float() \
                   * (prediction_error.detach() < 0.999).float() * (prediction_error.detach() > -0.999).float()
            acos_mask = acos_mask > 0.0
            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask]))
            logging_loss = torch.mean(torch.acos(prediction_error[acos_mask]))
            return optimize_loss, logging_loss

        elif mode == 'train_TAL_loss':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            # Robust acos loss
            acos_mask = mask.float() \
                   * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
            cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
            acos_mask = acos_mask > 0.0
            cos_mask = cos_mask > 0.0
            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask])
            logging_loss = optimize_loss.detach() / (torch.sum(cos_mask) + torch.sum(acos_mask))
            return optimize_loss, logging_loss

        elif mode == 'train_SR_only':
            prediction_error_g = torch.cosine_similarity(surface_normal_pred['I_g'], sample_batched['gravity'],
                                                       dim=1, eps=1e-6)
            prediction_error_a = torch.cosine_similarity(surface_normal_pred['I_a'], sample_batched['aligned_directions'],
                                                       dim=1, eps=1e-6)

            acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
            cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
            acos_mask_g = acos_mask_g > 0.0
            cos_mask_g = cos_mask_g > 0.0

            acos_mask_a = (prediction_error_a.detach() < 0.9999).float() * (prediction_error_a.detach() > 0.0).float()
            cos_mask_a = (prediction_error_a.detach() <= 0.0).float()
            acos_mask_a = acos_mask_a > 0.0
            cos_mask_a = cos_mask_a > 0.0

            optimize_loss = (torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])) \
                            + (torch.sum(torch.acos(prediction_error_a[acos_mask_a])) - torch.sum(prediction_error_a[cos_mask_a]))
            logging_loss = 0.5*(1.0-torch.mean(prediction_error_g) + 1.0-torch.mean(prediction_error_a))
            return optimize_loss, logging_loss

        elif mode == 'train_sr_fpn_full':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred['n'], sample_batched['Z'], dim=1, eps=1e-6)

            # Robust acos loss
            acos_mask = mask.float() \
                        * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
            cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
            acos_mask = acos_mask > 0.0
            cos_mask = cos_mask > 0.0

            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask])
            logging_loss = optimize_loss.detach() / (torch.sum(cos_mask) + torch.sum(acos_mask))

            rectifier_loss_list = [i for i, split in enumerate(sample_batched['ga_split']) if split != 'no_ga']

            if len(rectifier_loss_list) > 0:
                rectifier_loss_list = torch.LongTensor(rectifier_loss_list)
                I_g = surface_normal_pred['I_g'][rectifier_loss_list]
                I_a = surface_normal_pred['I_a'][rectifier_loss_list]
                gravity_dir_gt = sample_batched['gravity'][rectifier_loss_list]
                aligned_dir_gt = sample_batched['aligned_directions'][rectifier_loss_list]

                prediction_error_g = torch.cosine_similarity(I_g, gravity_dir_gt, dim=1, eps=1e-6)
                prediction_error_a = torch.cosine_similarity(I_a, aligned_dir_gt, dim=1, eps=1e-6)

                acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
                cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
                acos_mask_g = acos_mask_g > 0.0
                cos_mask_g = cos_mask_g > 0.0

                acos_mask_a = (prediction_error_a.detach() < 0.9999).float() * (prediction_error_a.detach() > 0.0).float()
                cos_mask_a = (prediction_error_a.detach() <= 0.0).float()
                acos_mask_a = acos_mask_a > 0.0
                cos_mask_a = cos_mask_a > 0.0

                optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask]) + \
                                76800 * (torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])) +\
                                76800 * (torch.sum(torch.acos(prediction_error_a[acos_mask_a])) - torch.sum(prediction_error_a[cos_mask_a]))
                logging_loss = 0.5*(1.0-torch.mean(prediction_error_g) + 1.0-torch.mean(prediction_error_a))

            return optimize_loss, logging_loss
