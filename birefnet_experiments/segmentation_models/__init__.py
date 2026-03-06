"""Unified segmentation model adapters used by benchmark scripts."""

from .base import SegmentationModelAdapter
from .birefnet_adapter import BiRefNetAdapter
from .optional_adapters import U2NetAdapter, MODNetAdapter, SAMAdapter, ISNetAdapter


def create_model(name: str, **kwargs):
    name = name.lower()
    if name == "birefnet":
        return BiRefNetAdapter(**kwargs)
    if name == "u2net":
        return U2NetAdapter(**kwargs)
    if name == "modnet":
        return MODNetAdapter(**kwargs)
    if name == "sam":
        return SAMAdapter(**kwargs)
    if name == "isnet":
        return ISNetAdapter(**kwargs)
    raise ValueError(f"Unknown model '{name}'.")
