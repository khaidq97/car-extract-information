import logging
import os

def get_logger(name, save_dir, mode='debug'):
    if mode == 'info':
        mode_ == logging.DEBUG
    else:
        mode_ = logging.DEBUG
        
    logger = logging.getLogger(name)
    logger.setLevel(mode_)
    # create console handler and set level to debug
    ch = logging.StreamHandler()

    # create file handler and set level to error
    fh = logging.FileHandler(os.path.join(save_dir, 'log.log'))

    # create formatter
    formatter = logging.Formatter('%(message)s')

    # add formatter to handlers
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger