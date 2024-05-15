# change lr0:0.001 in data/hyp.scratch.tiny.yaml
CUDA_VISIBLE_DEVICES=0 python train.py \
  --epochs 20 \
  --workers 2 \
  --device 0 \
  --batch-size 32 \
  --data data/scuthead.yaml \
  --img 640 640 \
  --cfg cfg/yolov7.yaml \
  --name yolov7-finetune-scuthead-head \
  --hyp data/hyp.scratch.tiny.yaml \
  --weight weights/pretrain-coco.pt \
  --multilosses True \
  --detect-layer 'IDetectHead' \
  --freeze '106,107-108'