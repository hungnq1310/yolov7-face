python detect.py \
    --weights weights/pretrain-coco.pt \
    --device 0 \
    --source data/images \
    --detect-layer body \
    --save-txt \
    --kpt-label 17 \
    --conf-thres 0.4 \
    --name  \
    --exist-ok