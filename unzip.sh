pushd model
if [ ! -f "detsub.onnx" ]; then
    tar xvf detsub.onnx.tar.xz
fi

if [ ! -f "rec.onnx" ]; then
    tar xvf rec.onnx.tar.xz
fi
popd