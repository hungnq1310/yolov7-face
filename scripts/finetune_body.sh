# change lr0:0.001 in data/hyp.scratch.tiny.yaml

CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 20 \
  --workers 2 \
  --device 0 \
  --batch-size 32 \
  --data data/body_pose.yaml \
  --img 640 640 \
  --cfg cfg/yolov7.yaml \
  --name yolov7-finetune-coco-body \
  --hyp data/hyp.scratch.tiny.yaml \
  --weight weights/pretrain-coco.pt \
  --multilosses True \
  --detect-layer 'IKeypointBody' \
  --freeze '105,106-108' \
  --kpt-label 17