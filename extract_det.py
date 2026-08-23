import onnx

def main(src_path, dst_path):
    input_names = ["x"]
    output_names = ["conv2d_213.tmp_0"]
    onnx.utils.extract_model(src_path, dst_path, input_names, output_names)

main("model/det.onnx", "model/detsub.onnx")
