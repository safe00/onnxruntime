import numpy as np

import onnxruntime as ort

np.random.seed(0)

ROOT_MODEL_PATH = "/mnt/nfs/jenkins_iotsw/hhb/tools/ai-bench/net/"


def cos_sim(a, b):
    a = np.reshape(a, [-1]).astype("float32")
    b = np.reshape(b, [-1]).astype("float32")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    dot_product = np.dot(a, b)
    cos_sim = dot_product / (norm_a * norm_b)
    return cos_sim


def conver_data(input_dict, target_dtype):
    out = {}
    for x in input_dict:
        if input_dict[x].dtype == "float32":
            out[x] = input_dict[x].astype(target_dtype)
        else:
            out[x] = input_dict[x]
    return out


def test_model(model_path, input_dict, is_fp16=False, base_api="CSINN_C920", equal_th=0.99):
    times = 2

    if is_fp16:
        inputs = conver_data(input_dict, "float16")
    else:
        inputs = input_dict

    shl_session = ort.InferenceSession(
        model_path,
        providers=["ShlExecutionProvider"],
        provider_options=[
            {
                "base_api": base_api,
                # "debug_level": "CSINN_DEBUG_LEVEL_DEBUG",
                # "profiler_level": "CSINN_PROFILER_LEVEL_DUMP",
            },
        ],
    )
    for _ in range(times):
        shl_output = shl_session.run(None, inputs)

    cpu_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    output = cpu_session.run(None, inputs)

    for x, y in zip(shl_output, output):
        # if not is_fp16:
        #     np.testing.assert_array_almost_equal(x, y, 3)
        assert cos_sim(x, y) >= equal_th


def test_effcientnet():
    in_name = ["data"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/efficientnet/efficientnet_b0.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "_fp16.onnx"), input_dict, is_fp16=True)


def test_resnet():
    in_name = ["data"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/resnet/resnet50-v1-7.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "-fp16.onnx"), input_dict, is_fp16=True)
    test_model(model_path.replace(".onnx", "-qdq.onnx"), input_dict, base_api="CSINN_TH1520")


def test_mv2():
    in_name = ["input"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/mobilenet/mobilenetv2-12.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "_fp16.onnx"), input_dict, is_fp16=True)
    test_model(model_path.replace(".onnx", "-sim-qdq.onnx"), input_dict, base_api="CSINN_TH1520", equal_th=0.9479)


def test_shufflenet():
    in_name = ["input"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/shufflenet/shufflenet-v2-10.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "-fp16.onnx"), input_dict, is_fp16=True)


def test_swin():
    in_name = ["input"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/swin/swin_tiny_v1_export_sim.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "_fp16.onnx"), input_dict, is_fp16=True)


def test_retinaface():
    in_name = ["input0"]
    in_shape = [[1, 3, 640, 640]]
    model_path = ROOT_MODEL_PATH + "onnx/retinaface/models/FaceDetector.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "_fp16.onnx"), input_dict, is_fp16=True)
    test_model(model_path.replace(".onnx", "_qdq.onnx"), input_dict, base_api="CSINN_TH1520", equal_th=0.9412)


def test_mobilenet_vit():
    in_name = ["input"]
    in_shape = [[1, 3, 256, 256]]
    model_path = ROOT_MODEL_PATH + "onnx//mobile-vit/mobilevit_v1_small_export_sim.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "_fp16.onnx"), input_dict, is_fp16=True)


def test_yolov5n():
    in_name = ["images"]
    in_shape = [[1, 3, 640, 640]]
    model_path = ROOT_MODEL_PATH + "onnx/yolo/yolov5n_cut_fp32.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)
    test_model(model_path.replace("_fp32.onnx", "_fp16.onnx"), input_dict, is_fp16=True)
    test_model(model_path.replace("_fp32.onnx", "_qdq.onnx"), input_dict, base_api="CSINN_TH1520")


def test_bert():
    in_name = ["input_ids", "input_mask", "segment_ids"]
    in_shape = [[1, 384], [1, 384], [1, 384]]
    model_path = ROOT_MODEL_PATH + "onnx/bert_end2end/models/bert_small/bert_small_int32_input_no_batch.onnx"
    input_data = [[], [], []]
    input_data[0] = (
        np.fromfile("/mnt/nfs/jenkins_iotsw/hhb/flow/data_sub/bert_small/sample_1_input_ids.bin", dtype="float32")
        .reshape(in_shape[0])
        .astype("int32")
    )
    input_data[1] = (
        np.fromfile("/mnt/nfs/jenkins_iotsw/hhb/flow/data_sub/bert_small/sample_1_input_mask.bin", dtype="float32")
        .reshape(in_shape[1])
        .astype("int32")
    )
    input_data[2] = (
        np.fromfile("/mnt/nfs/jenkins_iotsw/hhb/flow/data_sub/bert_small/sample_1_segment_ids.bin", dtype="float32")
        .reshape(in_shape[2])
        .astype("int32")
    )
    input_dict = {
        in_name[0]: input_data[0],
        in_name[1]: input_data[1],
        in_name[2]: input_data[2],
    }
    test_model(model_path, input_dict)
    test_model(model_path.replace(".onnx", "_fp16.onnx"), input_dict, is_fp16=True)


if __name__ == "__main__":
    test_effcientnet()
    test_resnet()
    test_mv2()
    test_retinaface()
    test_swin()
    test_shufflenet()
    test_yolov5n()
    test_bert()
    test_mobilenet_vit()
