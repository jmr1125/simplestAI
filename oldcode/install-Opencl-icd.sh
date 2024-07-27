echo for more infomation : visit https://github.com/KhronosGroup/OpenCL-ICD-Loader 
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
git clone https://github.com/KhronosGroup/OpenCL-Headers

cmake -D CMAKE_INSTALL_PREFIX=./OpenCL-Headers/install -S ./OpenCL-Headers -B ./OpenCL-Headers/build 
cmake --build ./OpenCL-Headers/build --target install

cmake -D CMAKE_PREFIX_PATH=`pwd`/OpenCL-Headers/install -D CMAKE_INSTALL_PREFIX=./OpenCL-ICD-Loader/install -S ./OpenCL-ICD-Loader -B ./OpenCL-ICD-Loader/build 
cmake --build ./OpenCL-ICD-Loader/build --target install
