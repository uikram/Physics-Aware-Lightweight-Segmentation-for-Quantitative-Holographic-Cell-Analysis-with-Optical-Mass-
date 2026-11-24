# Pretrained weights

Weights are not part of the repository. Place these three files here (the names are set in `configs/config.yaml`,
`models.<arch>.weights`):

| File | Used by | Source |
|---|---|---|
| `edge_sam_3x.pth` | EdgeSAM (RepViT-M1 image encoder; only the `image_encoder.*` tensors are loaded, strictly) | [EdgeSAM on Hugging Face](https://huggingface.co/chongzhou/EdgeSAM/resolve/main/edge_sam_3x.pth) · [EdgeSAM repository](https://github.com/chongzhou96/EdgeSAM) |
| `mobile_sam.pt` | MobileSAM (TinyViT image encoder) | [`weights/mobile_sam.pt` in the MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt) |
| `mobilenet_v2-b0353104.pth` | MobileNet-UNet (ImageNet MobileNetV2 encoder) | torchvision `MobileNet_V2_Weights.IMAGENET1K_V1`: <https://download.pytorch.org/models/mobilenet_v2-b0353104.pth> |

```bash
wget -P weights https://huggingface.co/chongzhou/EdgeSAM/resolve/main/edge_sam_3x.pth
wget -P weights https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget -P weights https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
```

Only the encoders are initialised from these files. The decoders are initialised randomly and trained on the
phase data, and the three-channel input convolution is replaced by a single-channel one whose filters are the mean
of the pretrained RGB filters.

Trained checkpoints (`training_results/<Arch>/<condition>/<loss>/seed<s>/checkpoints/{last,best}.pt`) are not
distributed either.

The weights are covered by the licences of their projects: EdgeSAM is under the S-Lab License 1.0
(non-commercial), MobileSAM under Apache-2.0, and torchvision under BSD-3-Clause.
