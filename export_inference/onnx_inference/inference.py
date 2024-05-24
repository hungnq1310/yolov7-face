import os
import numpy as np
import cv2
import argparse
import onnxruntime
import glob

from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="./yolo.onnx")
parser.add_argument("--img-path", type=str, default="./data/images/22_Picnic_Picnic_22_10.jpg")
parser.add_argument("--video-path", type=str, default="./data/video/test_2.mp4")
parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
parser.add_argument("--get-layer", type=str, default="head")
parser.add_argument("--head-thres", type=float, default=0.75)
parser.add_argument("--face-thres", type=float, default=0.78)
parser.add_argument("--body-thres", type=float, default=0.7)
args = parser.parse_args()


sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(args.model_path, sess_options=sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

_CLASS_COLOR_MAP = [
    (0, 0, 255) , # Person (blue).
    (255, 0, 0) ,  # Bear (red).
    (0, 255, 0) ,  # Tree (lime).
    (255, 0, 255) ,  # Bird (fuchsia).
    (0, 255, 255) ,  # Sky (aqua).
    (255, 255, 0) ,  # Cat (yellow).
]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def postprocess(img, dst_file, output, dwdh=0, score_threshold=0.3, ratio=0, get_layer=None):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    # img = cv2.imread(img_file)

    padding = dwdh*2
    det_bboxes, det_scores, det_labels  = output[:, 1:5], output[:, 6], output[:, 5]

    if get_layer == 'face' or get_layer == 'body':
        kpts = output[:, 7:]

    for idx in range(len(det_bboxes)):
        color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]
        det_bbox = det_bboxes[idx]
        det_bbox -= np.array(padding)
        det_bbox /= ratio
        det_bbox = det_bbox.round().astype(np.int32).tolist()

        # get class_id & score
        # cls_id = int(det_labels[idx])
        score = round(float(det_scores[idx]),3)

        # draw bbox
        if get_layer == 'face':
            name = 'face'
        elif get_layer == 'head':
            name = 'head'
        else:
            name = 'body'

        if score > score_threshold:
            cv2.rectangle(img, det_bbox[:2],det_bbox[2:],color_map[::-1],2)
            cv2.putText(img, str(score), (det_bbox[0], det_bbox[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color_map[::-1],thickness=2)

        # draw keypoints
        if get_layer == 'face' or get_layer == 'body':
            plot_skeleton_kpts(img, kpts[idx], pdg=padding, ratio=ratio, steps=3, radius=2)
            
    return img
    # cv2.imwrite(str(dst_file), img)

def plot_skeleton_kpts(im, kpts, pdg=0, ratio=0, steps=3, radius=2):
    num_kpts = len(kpts) // steps
    
    l_pair = [[16, 14], [14, 12], [17, 15], 
                [15, 13], [12, 13], [6, 12], 
                [7, 13], [6, 7], [6, 8], 
                [7, 9], [8, 10], [9, 11], 
                [2, 3], [1, 2], [1, 3], 
                [2, 4], [3, 5], [4, 6], [5, 7]]
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
              (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]

    part_line = {}
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid] , kpts[steps * kid + 1]

        x_coord -= np.array(pdg[0])
        y_coord -= np.array(pdg[1])

        conf = kpts[steps * kid + 2]

        if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
            part_line[kid+1] = (int(x_coord/ratio), int(y_coord/ratio))
            cv2.circle(im, (int(x_coord/ratio), int(y_coord/ratio)), radius, (int(r), int(g), int(b)), -1)
            # cv2.putText(im, str(conf), (int(x_coord/ratio), int(y_coord/ratio)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
                
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            if i < len(line_color):
                cv2.line(im, start_xy, end_xy, line_color[i], 2)
            else:
                cv2.line(im, start_xy, end_xy, (255,255,255), 1)
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def read_img(img_file):
    image = cv2.imread(img_file)[:, :, ::-1]
    image, ratio, dwdh = letterbox(image, auto=False)
    
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    image /= 255

    return image, ratio, dwdh

def model_inference(inp=None):
    #onnx_model = onnx.load(args.model_path)
    input_name = session.get_inputs()[0].name
    if isinstance(inp, list):
        inp = np.stack(inp).squeeze()
    output = session.run([], {input_name: inp}) # {input_name: [inp, inp, inp]} 
    return output           


def video_inference(model_path, video_path, dst_path, get_layer=None):
    # os.makedirs(dst_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(dst_path, 
                          cv2.VideoWriter_fourcc(*'h264'), 
                          fps, (frame_width, frame_height))
    print(video_path, dst_path)
    print("-"*20)
    for i in tqdm(range(nframes)):
        ret, frame = cap.read()
        if not ret:
            break

        image, ratio, dwdh = letterbox(frame, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32)
        image /= 255

        pred = model_inference(image)

        if args.get_layer == 'head':
            output = pred[0]
            score_thres = args.head_thres
        elif args.get_layer == 'face':
            output = pred[1] 
            score_thres = args.face_thres
        elif args.get_layer == 'body':
            output = pred[2] 
            score_thres = args.body_thres
        else:
            raise NotImplementedError

        frame = postprocess(frame, dst_path, output, dwdh, score_threshold=score_thres, ratio=ratio, get_layer=get_layer)
        out.write(frame)
    cap.release()
    out.release()
            
def model_inference_image_list(model_path, img_path=None, dst_path=None, get_layer=None):
    os.makedirs(args.dst_path, exist_ok=True)

    img_formats = [ "bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo", ]
    p = str(Path(img_path).absolute())  # os-agnostic absolute path
    if "*" in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f"ERROR: {p} does not exist")

    img_file_list = [x for x in files if x.split(".")[-1].lower() in img_formats]
   
    pbar = enumerate(img_file_list)
    max_index = 25
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))
    for img_index, img_file  in pbar:
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        img_file = img_file.rstrip()
        image, ratio, dwdh = read_img(img_file)
        pred = model_inference(model_path, image)
        
        if args.get_layer == 'head':
            output = pred[0]
            score_thres = args.head_thres
        elif args.get_layer == 'face':
            output = pred[1] 
            score_thres = args.face_thres
        elif args.get_layer == 'body':
            output = pred[2] 
            score_thres = args.body_thres
        else:
            raise NotImplementedError
        
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        postprocess(img_file, dst_file, output, dwdh, score_threshold=score_thres, ratio=ratio, get_layer=get_layer)


def main():
    # model_inference_image_list(model_path=args.model_path, img_path=args.img_path,
    #                            dst_path=args.dst_path,
    #                            get_layer=args.get_layer)

    video_inference(
        model_path=args.model_path, 
        video_path=args.video_path,
        dst_path=args.dst_path,
        get_layer=args.get_layer
    )

if __name__== "__main__":
    main()