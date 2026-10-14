#!/bin/sh
set -eu
cd "$(dirname "$0")"
revision=a341b4925024b88b2c593468f16e12f5e17315da
mkdir -p .native
if [ ! -d .native/mlx-c/.git ]; then
  git clone https://github.com/ml-explore/mlx-c.git .native/mlx-c
fi
if [ "$(git -C .native/mlx-c rev-parse HEAD)" != "$revision" ]; then
  git -C .native/mlx-c fetch origin "$revision"
  git -C .native/mlx-c checkout --detach "$revision"
fi
cmake -S .native/mlx-c -B .native/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DMLX_C_BUILD_EXAMPLES=OFF -DMLX_BUILD_METAL=ON
cmake --build .native/build -j "${CL_MLX_BUILD_JOBS:-4}"
cmake -S native -B .native/shim -DCMAKE_BUILD_TYPE=Release \
  -DMLX_C_SOURCE="$PWD/.native/mlx-c" -DMLX_C_BUILD="$PWD/.native/build"
cmake --build .native/shim
