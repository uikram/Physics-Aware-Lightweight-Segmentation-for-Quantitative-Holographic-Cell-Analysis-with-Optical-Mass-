# Pretrained backbone weights

The models are initialised from the official pretrained checkpoints below. They are
third-party files and are not tracked in this repository. Download them here:

| File | Used by | Source |
|---|---|---|
| `edge_sam_3x.pth` | `models/edge_sam.py` | [EdgeSAM](https://github.com/chongzhou/EdgeSAM) (Hugging Face: `chongzhou/EdgeSAM`) |
| `mobile_sam.pt` | `models/mobile_sam.py` | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |

```bash
wget -P weights https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth
wget -P weights https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

MobileNet-UNet uses torchvision's ImageNet `MobileNet_V2_Weights.IMAGENET1K_V1`, which is downloaded
automatically into `./cache`.

Fine-tuned checkpoints are written to `results/<run>/checkpoints/best_model.pt` during training.
