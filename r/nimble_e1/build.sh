#!/usr/bin/env bash
# Build a self-contained relocatable object (wrapper + static GSL members)
# so nimble can link it without extra linker flags.
set -euo pipefail
cd "$(dirname "$0")"
PREFIX="${GSL_PREFIX:-${CONDA_PREFIX:-$(dirname "$(dirname "$(command -v R)")")}}"
${CC:-gcc} -O2 -fPIC -c e1_gsl.c -I"$PREFIX/include" -o e1_gsl_raw.o
ld -r e1_gsl_raw.o "$PREFIX/lib/libgsl.a" -o e1_gsl.o
rm e1_gsl_raw.o
echo "built $(pwd)/e1_gsl.o"
