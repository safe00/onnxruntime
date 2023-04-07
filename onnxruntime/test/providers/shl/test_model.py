import numpy as np

import onnxruntime as ort

np.random.seed(0)

ROOT_MODEL_PATH = "/mnt/flow/iot/ai_tools/tools/ai-bench/net/"


def cos_sim(a, b):
    a = np.reshape(a, [-1])
    b = np.reshape(b, [-1])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 计算向量的点积
    dot_product = np.dot(a, b)

    # 计算余弦相似度
    cos_sim = dot_product / (norm_a * norm_b)
    return cos_sim


def test_model(model_path, input_dict):
    times = 3
    shl_session = ort.InferenceSession(model_path, providers=["ShlExecutionProvider"])
    for _ in range(times):
        shl_output = shl_session.run(None, input_dict)

    cpu_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for _ in range(times):
        output = cpu_session.run(None, input_dict)

    for x, y in zip(shl_output, output):
        np.testing.assert_array_almost_equal(x, y, 3)


def test_effcientnet():
    in_name = ["data"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/efficientnet/efficientnet_b0.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)


def test_resnet():
    in_name = ["data"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/resnet/resnet50-v1-7.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)


def test_mv2():
    in_name = ["input"]
    in_shape = [[1, 3, 224, 224]]
    model_path = ROOT_MODEL_PATH + "onnx/mobilenet/mobilenetv2-12.onnx"
    input_data = [np.random.random(x).astype("float32") for x in in_shape]
    input_dict = {in_name[0]: input_data[0]}
    test_model(model_path, input_dict)


if __name__ == "__main__":
    test_effcientnet()
    test_resnet()
    test_mv2()
