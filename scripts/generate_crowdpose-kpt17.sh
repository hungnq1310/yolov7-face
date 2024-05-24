
CUDA_VISIBLE_DEVICES=0 python third_party/mmpose/demo/inferencer_demo.py \
    ./datasets/crowdpose/images \
    --vis-out-dir ./predicts/vis_results/crowdpose \
    --pred-out-dir ./predicts/crowdpose-17kpt-mmengine \
    --pose2d third_party/mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py \
    --pose2d-weights ./weights/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth \
    --device 'cuda:0' \