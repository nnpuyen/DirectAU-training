# -*- coding: utf-8 -*-
# @Time   : 2020/8/7
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

# UPDATE
# @Time   : 2021/3/7
# @Author : Jiawei Guan
# @Email  : guanjw@ruc.edu.cn

"""
recbole.utils.logger
###############################
"""

import logging
import os
import colorlog
import torch

from recbole.utils.utils import get_local_time, ensure_dir
from colorama import init

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


def _get_runtime_gpu_info(config):
    """Collect GPU name and total memory at runtime for logging."""
    if not torch.cuda.is_available():
        return None, None

    try:
        gpu_id = int(config['gpu_id'])
    except Exception:
        gpu_id = 0

    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        gpu_id = 0

    prop = torch.cuda.get_device_properties(gpu_id)
    gpu_name = prop.name
    gpu_memory_total_mb = round(prop.total_memory / (1024 * 1024), 2)
    return gpu_name, gpu_memory_total_mb


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    logfilename = '{}-{}.log'.format(config['model'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
        
    # Use the plain formatter for the file handler
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    # Use the colored formatter for the stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)

    gpu_name, gpu_memory_total_mb = _get_runtime_gpu_info(config)
    if gpu_name is not None and gpu_memory_total_mb is not None:
        logger.info('GPU NAME: %s', gpu_name)
        logger.info('GPU MEMORY TOTAL MB: %s', gpu_memory_total_mb)

    return logger
