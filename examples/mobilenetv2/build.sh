#!/bin/bash

function print_help()
{
  echo "$0 -a <arch> "
  echo ""
  echo "    -a : arch (aarch64/musl_riscv64/glibc_riscv64)"
  echo "    -h : print help"
  echo "such as: $0 -a musl_riscv64"
  echo ""
  exit -1
}

set -e
# echo "$0 $@"
while getopts ":a:h" opt; do
  case $opt in
    a)
      TARGET_ARCH=$OPTARG
      ;;
    h)
      print_help
      ;;
    :)
      echo "Option -$OPTARG requires an argument." 
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG index:$OPTIND"
      print_help
      ;;
  esac
done

case ${TARGET_ARCH} in
  aarch64)
    TOOLCHAIN=toolchain-aarch64-linux.cmake
    ;;
  musl_riscv64)
    TOOLCHAIN=toolchain-riscv64-linux-musl-x86_64.cmake
    ;;
  glibc_riscv64)
    TOOLCHAIN=toolchain-riscv64-linux-x86_64.cmake
    ;;
  *)
    echo "Invalid Arch: ${TARGET_ARCH}"
    echo "Examples: $0 -a musl_riscv64"
    exit -1
    ;;
esac

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/install_${TARGET_ARCH}
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_ARCH}

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DTARGET_ARCH=${TARGET_ARCH} \
    -DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
    -DCMAKE_TOOLCHAIN_FILE=${ROOT_PWD}/../../toolchain/${TOOLCHAIN} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    ../..

cmake --build . --target install
