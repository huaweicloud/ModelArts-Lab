from functools import wraps


def enable_vae_lightning(self, return_output=False):
    self.vae.enable_lightning(return_output)
    self.video_processor.enable_lightning(return_output)


def enable_lightning(self, return_output=False):
    self.lightning = True
    self.return_output = return_output


def add_lightning_init(cls):
    orig_init = cls.__init__

    @wraps(orig_init)
    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.lightning = False
        self.return_output = True

    cls.__init__ = new_init
    return cls
