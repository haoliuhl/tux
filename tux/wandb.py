import os
import tempfile
import uuid
from socket import gethostname

import wandb
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder

from .config import flatten_config_dict, update_config_dict
from .utils import save_pickle


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.project_id = ""
        config.project_entity = placeholder(str)
        config.experiment_id = placeholder(str)
        config.append_uuid = True
        config.experiment_note = placeholder(str)

        config.output_dir = "/tmp/"
        config.wandb_dir = ""
        config.profile_dir = ""

        config.online = False

        return update_config_dict(config, updates)

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None or self.config.experiment_id == "":
            self.config.experiment_id = uuid.uuid4().hex
        else:
            if self.config.append_uuid:
                self.config.experiment_id = (
                    str(self.config.experiment_id) + "_" + uuid.uuid4().hex
                )
            else:
                self.config.experiment_id = str(self.config.experiment_id)

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                if not self.config.output_dir.startswith("gs://"):
                    os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.wandb_dir == "":
                if not self.config.output_dir.startswith("gs://"):
                    # Use the same directory as output_dir if it is not a GCS path.
                    self.config.wandb_dir = self.config.output_dir
                else:
                    # Otherwise, use a temporary directory.
                    self.config.wandb_dir = tempfile.mkdtemp()
            else:
                # Join the wandb_dir with the experiment_id.
                self.config.wandb_dir = os.path.join(
                    self.config.wandb_dir, self.config.experiment_id
                )
                os.makedirs(self.config.wandb_dir, exist_ok=True)

            if self.config.profile_dir == "":
                if not self.config.output_dir.startswith("gs://"):
                    # Use the same directory as output_dir if it is not a GCS path.
                    self.config.profile_dir = self.config.output_dir
                else:
                    # Otherwise, use a temporary directory.
                    self.config.profile_dir = tempfile.mkdtemp()
            else:
                # Join the profile_dir with the experiment_id.
                self.config.profile_dir = os.path.join(
                    self.config.profile_dir, self.config.experiment_id
                )
                os.makedirs(self.config.profile_dir, exist_ok=True)

        self._variant = flatten_config_dict(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project_id,
                dir=self.config.wandb_dir,
                id=self.config.experiment_id,
                resume="allow",
                notes=self.config.experiment_note,
                entity=self.config.project_entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        if self.enable:
            save_pickle(obj, os.path.join(self.config.output_dir, filename))

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir

    @property
    def wandb_dir(self):
        return self.config.wandb_dir

    @property
    def profile_dir(self):
        return self.config.profile_dir
