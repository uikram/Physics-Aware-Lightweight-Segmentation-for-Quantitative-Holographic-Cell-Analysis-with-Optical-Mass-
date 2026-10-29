"""Vendored RepViT image encoder of EdgeSAM (https://github.com/chongzhou96/EdgeSAM, commit d24d996).

Only ``rep_vit.py`` and ``common.py`` are copied, unmodified, so that the encoder can be built without the
rest of the ``edge_sam`` package (whose ``__init__`` imports mmdet). The upstream S-Lab License 1.0 (non-commercial)
applies to these files; see LICENSE in this directory.
"""
from .rep_vit import RepViT  # noqa: F401
