echo "🏗️ Building Seastar Dynamic Graphs"
echo " "

for graph in "$@"
do
    echo "🔨 Building $graph"
    cd dynamic/$graph/
    /usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC' -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o $graph.so $graph.cu
    echo "✅ $graph build completed"
    echo ""
    cd ../..
done