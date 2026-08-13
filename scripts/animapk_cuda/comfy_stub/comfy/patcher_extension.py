"""Stub of comfy.patcher_extension: no-op wrapper executor."""

from enum import Enum


class WrappersMP(Enum):
    DIFFUSION_MODEL = "diffusion_model"


class WrapperExecutor:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn

    def execute(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @classmethod
    def new_class_executor(cls, fn, self, wrappers):
        return WrapperExecutor(fn, self, wrappers)


def get_all_wrappers(wrapper_type, options=None):
    return []
