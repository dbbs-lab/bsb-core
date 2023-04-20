git clone https://github.com/nest/nest-simulator $GITHUB_WORKSPACE/nest
cd $GITHUB_WORKSPACE/nest
git checkout tags/v$1
mkdir build
cd build
pip install cython cmake
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$2 \
  -Dwith-mpi=ON
make install
