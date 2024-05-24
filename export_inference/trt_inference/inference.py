import os
import cv2
import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

from util import preprocess, scale_boxes, nms


class TRTYoloV7:
    def __init__(
        self,
        engine_path,
        warmup=True,
        nms_thresh=0.5,
        body_conf=0.4,
        head_conf=0.2,
        face_conf=0.4,
    ):
        self.names = ["head", "body", "face"]
        self.colors = {"head": (0, 255, 0), "body": (0, 0, 255), "face": (255, 0, 0)}
        self.nms_thr = nms_thresh
        self.body_conf = body_conf
        self.head_conf = head_conf
        self.face_conf = face_conf

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, "")  # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_tensor_shape(binding))
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

        if warmup:
            for _ in range(10):
                tmp = np.random.randn(1, 3, 640, 640).astype(np.float32)
                self.infer(tmp)

    def infer(self, img):
        self.inputs[0]["host"] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out["host"] for out in self.outputs]
        return data

    def predict(
        self,
        origin_img,
    ):
        img, ratio, dwdh = preprocess(origin_img)
        data = self.infer(img)
        (
            bnum_det,
            bboxes,
            bscores,
            bcategories,
            hnum_det,
            hboxes,
            hscores,
            hcategories,
            face,
        ) = data

        # Body
        bboxes = np.reshape(bboxes, (-1, 4))
        bboxes = bboxes[: bnum_det[0]]
        bscores = bscores[: bnum_det[0]]
        bcategories = bcategories[: bnum_det[0]]
        bboxes, bscores = scale_boxes(bboxes, bscores, ratio, dwdh, self.body_conf)

        # Head
        hboxes = np.reshape(hboxes, (-1, 4))
        hboxes = hboxes[: hnum_det[0]]
        hscores = hscores[: hnum_det[0]]
        hcategories = hcategories[: hnum_det[0]]
        hboxes, hscores = scale_boxes(hboxes, hscores, ratio, dwdh, self.head_conf)

        # Face ([0, 1, 2, 3], 4, 5, [6...20]) (box, score, class-conf, keypoints)
        face = np.reshape(face, (-1, 21))
        fboxes, fscores, flmks = self.face_postprocess(face)
        fboxes, fscores, flmks = scale_boxes(
            fboxes, fscores, ratio, dwdh, self.face_conf, flmks
        )

        return {
            "body": {
                "boxes": bboxes,
                "scores": bscores,
            },
            "head": {
                "boxes": hboxes,
                "scores": hscores,
            },
            "face": {
                "boxes": fboxes,
                "scores": fscores,
                "keypoints": flmks,
            },
            "original": origin_img,
            "ratio": ratio,
            "dwdh": dwdh,
        }

    def face_postprocess(self, faces, scores_thr=0.1):
        boxes = faces[:, :4]
        conf = faces[:, 4:5]
        scores = faces[:, 5:6]
        lmks = faces[:, 6:]

        scores *= conf

        # Filter out low scores
        mask = scores > scores_thr
        mask = mask.squeeze(1)
        boxes = boxes[mask]
        scores = scores[mask]
        lmks = lmks[mask]

        # XYWH to XYXY
        boxes[:, 2:] += boxes[:, :2]

        selected_indices = nms(boxes, scores.squeeze(1), nms_thr=self.nms_thr)

        selected_boxes = boxes[selected_indices]
        selected_scores = scores[selected_indices].squeeze()
        selected_lmks = lmks[selected_indices]

        w, h = (
            selected_boxes[:, 2] - selected_boxes[:, 0],
            selected_boxes[:, 3] - selected_boxes[:, 1],
        )
        dwdh = np.stack([w / 2, h / 2, w / 2, h / 2], 1)
        selected_boxes -= dwdh

        return selected_boxes, selected_scores, selected_lmks

    def visualize(self, image, dets, vis_class="body"):
        for box, score in zip(dets[vis_class]["boxes"], dets[vis_class]["scores"]):
            _box = box.astype(np.int32)
            name = vis_class
            color = self.colors[name]
            name += f" {score:.2f}"
            cv2.rectangle(image, _box[:2].tolist(), _box[2:].tolist(), color, 2)
            cv2.putText(
                image,
                name,
                (int(_box[0]), int(_box[1]) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness=2,
            )
            if vis_class == "face":
                for kpt in dets[vis_class]["keypoints"]:
                    for x, y in zip(kpt[::3], kpt[1::3]):
                        cv2.circle(image, (int(x), int(y)), 1, color, 2)
        return image


def parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="./input/22_Picnic_Picnic_22_36.jpg"
    )
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--weight", type=str, default="./weight/yolov7-tiny-v0.trt")
    parser.add_argument("--warmup", type=bool, default=True)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--body_conf", type=float, default=0.4)
    parser.add_argument("--head_conf", type=float, default=0.2)
    parser.add_argument("--face_conf", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    name = os.path.basename(args.input)
    model = TRTYoloV7(
        engine_path=args.weight,
        warmup=args.warmup,
        nms_thresh=args.nms_thresh,
        body_conf=args.body_conf,
        head_conf=args.head_conf,
        face_conf=args.face_conf,
    )

    image = cv2.imread(args.input)
    dets = model.predict(image)
    image = model.visualize(image, dets, vis_class="body")
    image = model.visualize(image, dets, vis_class="face")
    image = model.visualize(image, dets, vis_class="head")
    cv2.imwrite(os.path.join(args.output, name), image)
