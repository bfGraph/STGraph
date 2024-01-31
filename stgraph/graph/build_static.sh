echo "🔨 Building csr"
cd static/csr
/usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC' -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o csr.so csr.cu
echo "✅ csr build completed"
cd ../..
echo ""