# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np
import scipy.misc

if sys.version_info.major == 2:  # Python 2.x
    from StringIO import StringIO as BIO
else:  # Python 3.x
    from io import BytesIO as BIO

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, logstr="logdir"):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        self.tensorboard_dir = self.log_dir / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )
        self.logger_path = self.log_dir / "{:}.log".format(logstr)
        self.logger_file = open(self.logger_path, "w")

        self.writer = SummaryWriter(self.tensorboard_dir)

    def __repr__(self):
        return "{name}(dir={log_dir}, use-tf={use_tf}, writer={writer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def path(self, mode):
        if mode == "log":
            return self.log_dir
        else:
            raise TypeError("Unknow mode = {:}".format(mode))

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write("\r{%s}" % string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write("{:}\n".format(string))
            self.logger_file.flush()

    def scalar_summary(self, tags, values, step):
        """Log a scalar variable."""
        assert isinstance(tags, list) == isinstance(
            values, list
        ), "Type : {:} vs {:}".format(type(tags), type(values))
        if not isinstance(tags, list):
            tags, values = [tags], [values]

        for tag, value in zip(tags, values):
            self.writer.add_scalar(tag, value, step + 1)
