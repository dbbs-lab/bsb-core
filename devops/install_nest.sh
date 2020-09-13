echo ""
echo "Empty cache, starting NEST v$NEST_VERSION installation."
echo ""
export MY_BEFORE_DIR=$PWD
cd /home/travis
wget https://github.com/nest/nest-simulator/archive/v$NEST_VERSION.tar.gz -O nest-simulator-$NEST_VERSION.tar.gz
tar -xzf nest-simulator-$NEST_VERSION.tar.gz
mkdir nest-simulator-$NEST_VERSION-build
mkdir nest-install-$NEST_VERSION
cd nest-simulator-$NEST_VERSION-build
cmake \
  -Dwith-python=3 \
  -DPYTHON_EXECUTABLE=/home/travis/virtualenv/python3.8.1/bin/python3 \
  -DPYTHON_LIBRARY=/opt/python/3.8.1/lib/libpython3.8m.so \
  -DPYTHON_INCLUDE_DIR=/opt/python/3.8.1/include/python3.8m/ \
  -DCMAKE_INSTALL_PREFIX:PATH=/home/travis/nest-$NEST_VERSION \
  /home/travis/nest-simulator-$NEST_VERSION
make
make install
cd $MY_BEFORE_DIR
