paddle2onnx --model_dir ../inference/ch_PP-OCRv4_det_server_infer \
            --model_filename inference.pdmodel --params_filename inference.pdiparams \
            --save_file model/det.onnx \
            --enable_onnx_checker True
paddle2onnx --model_dir ../inference/ch_PP-OCRv4_rec_server_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file model/rec.onnx \
            --enable_onnx_checker True
