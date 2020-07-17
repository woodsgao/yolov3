import argparse
import logging
import os
import os.path as osp
import sys

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

# from pytorch_modules.tasks import train
from pytorch_modules.engine import build_modules, load_obj


def run(cfg):
    print(cfg.pretty())
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)
    checkpoint_callback = build_modules(cfg.checkpoint)
    model = load_obj(cfg.model.model_name)(cfg)
    callbacks = build_modules(cfg.callbacks)
    trainer = pl.Trainer(logger=tb_logger,
                         checkpoint_callback=checkpoint_callback,
                         callbacks=callbacks,
                         **cfg.trainer)
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default='conf/config.yaml',
                        type=str,
                        help='Your config file path.')
    parser.add_argument('--strict',
                        action='store_true',
                        help='Strict mode for hydra.')

    opt, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left
    hydra_wrapper = hydra.main(config_path=opt.config_path, strict=opt.strict)
    hydra_wrapper(run)()
