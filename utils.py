import torch
import numpy as np


def log(str, fp=None):
    if fp is not None:
        fp.write('%s\n' % (str))
        fp.flush()
    print(str)


def check_nan_ckpt(cnn):
    is_nan_flag = False
    for name, param in cnn.named_parameters():
        if torch.sum(torch.isnan(param.data)):
            is_nan_flag = True
            break
    return is_nan_flag

def log_normal_stats(epoch, iter, normal_error_in_angle, fp=None):
    log('Epoch %d, Iter %d, Mean %f, Median %f, Rmse %f, 5deg %f, 7.5deg %f, 11.25deg %f, 22.5deg %f, 30deg %f' %
    (epoch, iter, np.average(normal_error_in_angle), np.median(normal_error_in_angle),
     np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
     np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]), fp)

    print('%f %f %f %f %f %f %f %f' %
        (np.average(normal_error_in_angle), np.median(normal_error_in_angle),
        np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
        np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]))
