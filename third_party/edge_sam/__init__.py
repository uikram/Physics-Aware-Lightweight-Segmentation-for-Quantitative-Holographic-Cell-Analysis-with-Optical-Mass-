"""RepViT image encoder of EdgeSAM, copied from the official repository
https://github.com/chongzhou96/EdgeSAM (commit d24d99671f41a9c0003061248bded64a481e9059, 24 May 2025):
edge_sam/modeling/rep_vit.py and common.py. The only change is the timm import path (try the new location first),
needed for timm >= 0.9; the classes are the same. Only these two files are taken because the package's own
__init__ imports mmdet. License: S-Lab License 1.0 (non-commercial), see LICENSE."""
from .rep_vit import RepViT  # noqa: F401
