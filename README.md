# Rotated-Oriented

## Abstract
Rotated object detection is a field computer
vision and image processing that aims to identifying objects
in images when those objects can rotate freely. Many
algorithms have been released and achieved good results
on DOTA dataset. In this research, we conduct experiments
some methods such as Rotated RetinatNet-OBB/HBB,
Rotated FasterRCNN-OBB, Rotated RepPoints-OBB, Ori-
ented RepPoints, Oriented R-CNN on our dataset DOTA-
bus. The results show that some methods achieve good
mAP, such as Rotated FasterRCNN-OBB (60...), and aver-
age fluctuations between 40 and 50 mAP.

## Results and Models

RetinaNet

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) |  Aug  | Batch Size |                                                  Configs                                                  |                                                                                                                                                                                            Download                                                                                                                                                                                            |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :---: | :--------: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 64.55 |  oc   |   1x    |   3.38   |      15.7      |   -   |     2      |         [rotated_retinanet_hbb_r50_fpn_1x_dota_oc](./model/Rotated RetinaNet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/custom_rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py)         |                 [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc_20220121_095315.log.json)                 |
| ResNet50 (1024,1024,200) | 68.42 | le90  |   1x    |   3.38   |      16.9      |   -   |     2      |       [rotated_retinanet_obb_r50_fpn_1x_dota_le90](./rotated_retinanet_obb_r50_fpn_1x_dota_le90.py)       |             [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90_20220128_130740.log.json)             |
| ResNet50 (1024,1024,200) | 68.79 | le90  |   1x    |   2.36   |      22.4      |   -   |     2      |  [rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90](./rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py)  |   [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90_20220303_183714.log.json)   |
| ResNet50 (1024,1024,200) | 69.79 | le135 |   1x    |   3.38   |      17.2      |   -   |     2      |      [rotated_retinanet_obb_r50_fpn_1x_dota_le135](./rotated_retinanet_obb_r50_fpn_1x_dota_le135.py)      |           [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135_20220128_130755.log.json)           |
| ResNet50 (1024,1024,500) | 76.50 | le90  |   1x    |          |      17.5      | MS+RR |     2      | [rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90](./rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90_20220210_114843.log.json) |
