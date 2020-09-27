echo ""
echo "Empty cache, starting NEST v$NEST_VERSION installation."
echo ""
export MY_BEFORE_DIR=$PWD
cd /home/travis
git clone --depth 1 https://github.com/dbbs-lab/nest-simulator nest-simulator-$NEST_VERSION
mkdir nest-simulator-$NEST_VERSION-build
mkdir nest-install-$NEST_VERSION
cd nest-simulator-$NEST_VERSION-build
python -m pip install cython --upgrade
PYTHON_INCLUDE_DIR=`python3 -c "import sysconfig; print(sysconfig.get_path('include'))"`
echo "Include dir: $PYTHON_INCLUDE_DIR"
PYLIB_BASE=lib`basename $PYTHON_INCLUDE_DIR`
echo "Pylib base: $PYLIB_BASE"
PYLIB_DIR=$(dirname `sed 's/include/lib/' <<< $PYTHON_INCLUDE_DIR`)
echo "Pylib dir: $PYLIB_DIR"
PYTHON_LIBRARY=`find $PYLIB_DIR \( -name $PYLIB_BASE.so -o -name $PYLIB_BASE.dylib \) -print -quit`
PYTHON_LIBRARY=/opt/python/3.8.1/lib/libpython3.so
echo "--> Detected PYTHON_LIBRARY=$PYTHON_LIBRARY"
echo "--> Detected PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR"
CONFIGURE_PYTHON="-DPYTHON_LIBRARY=$PYTHON_LIBRARY -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR"
cmake \
    -DCMAKE_INSTALL_PREFIX:PATH=/home/travis/nest-$NEST_VERSION \
    -Dwith-mpi=ON \
    $CONFIGURE_PYTHON \
    /home/travis/nest-simulator-$NEST_VERSION
make
make install
cd $MY_BEFORE_DIR
