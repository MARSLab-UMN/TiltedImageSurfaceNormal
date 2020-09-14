import torch
import numpy as np
import skimage.io as sio
import cv2


def saving_rgb_tensor_to_file(rgb_tensor, path):
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255)
    sio.imsave(path, output_rgb_img)


def saving_mask_tensor_to_file(mask_tensor, path):
    output_rgb_img = np.uint8((mask_tensor.detach().cpu()) * 255)
    sio.imsave(path, output_rgb_img)


def saving_normal_tensor_to_file(normal_tensor, path, mode='L2'):
    normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=0)
    output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)
    sio.imsave(path, output_normal_img)


def construct_error_img(angle_error_img, valid_mask_img):
    angle_error_img = angle_error_img.detach().cpu().numpy()
    valid_mask_img = valid_mask_img.detach().cpu().numpy()
    valid_mask_img = valid_mask_img > 0

    error_mask_pi_4 = angle_error_img >= 45.
    angle_error_img[error_mask_pi_4] = 45.

    output_error_img = cv2.applyColorMap(np.uint8(angle_error_img * 255 / 45.), cv2.COLORMAP_JET)
    output_error_img = cv2.cvtColor(output_error_img, cv2.COLOR_RGB2BGR)
    output_error_img[:, :, 0][~valid_mask_img] = 128
    output_error_img[:, :, 1][~valid_mask_img] = 128
    output_error_img[:, :, 2][~valid_mask_img] = 128
    return output_error_img