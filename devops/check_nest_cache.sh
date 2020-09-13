if [ ! -d /home/travis/nest-$NEST_VERSION/lib/python3.8 ] ; then
  export HAS_NEST_CACHE=0
else
  export HAS_NEST_CACHE=1
fi
if [ ! -f /home/travis/nest-$NEST_VERSION/lib/nest/libcerebmodule.so ] ; then
  export HAS_CEREBNEST_CACHE=0
else
  export HAS_CEREBNEST_CACHE=1
fi
