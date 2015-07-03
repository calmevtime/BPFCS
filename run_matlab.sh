#!/bin/sh
export LIB_GCC=/usr/lib/gcc/x86_64-linux-gnu/4.8/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/gcc/x86_64-linux-gnu/4.8/:/
export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so:$LIB_GCC/libstdc++.so:$LIB_GCC/libgomp.so
C: