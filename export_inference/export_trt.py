import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )
        # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        print("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            print(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size  # This no effect for networks created with explicit batch dimension mode. Also DEPRECATED.

    def create_engine(
        self,
        engine_path,
        precision,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        _ = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        with self.builder.build_serialized_network(
            self.network, self.config
        ) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()


def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16"],
        help="The precision mode to build in, either 'fp32', 'fp16', default: 'fp16'",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=1,
        type=int,
        help="The max memory workspace size to allow in Gb, " "default: 1",
    )
    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)

    main(args)
