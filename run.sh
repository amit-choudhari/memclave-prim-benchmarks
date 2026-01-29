#!/bin/bash
set -euo pipefail

# basic PrIM benchmark suite of 16 benchmarks
RUN_PRIM="no"

# BFS and MLP benchmarks with multiple inputs
RUN_BFSMLP="no"

# Benchmark evaluating MRAM throughput
RUN_MRAM="no"

# Output Directory of CSV Files
OUTDIR=$(realpath output/)

function print_help_exit {
  echo "Usage: ./run.sh ([ all | prim | mlpbfs | fast | mram ])?"
  echo "Run benchmarks within the UPMEM execution environment. Usually no command line option is required."
  echo "You may alter the number of benchmarks executed by passing one of the following parameters:"
  echo "all    - Run all benchmarks (default)"
  echo "fast   - Run default benchmarks except for the MRAM one"
  echo "prim   - Run the full PrIM benchmark suite with default input sizes"
  echo "bfsmlp - Run the MLP and BFS benchmarks with different input sizes"
  echo "mram   - Run the MRAM benchmark"
  exit 1
}

if [ "$#" == "0" ];
then
  RUN_PRIM="yes"
  RUN_BFSMLP="yes"
  RUN_MRAM="yes"
elif [ "$#" == "1" ];
then
  if [ "$1" == "all" ];
  then
    RUN_PRIM="yes"
    RUN_BFSMLP="yes"
    RUN_MRAM="yes"
  elif [ "$1" == "fast" ];
  then
    RUN_PRIM="yes"
    RUN_BFSMLP="yes"
  elif [ "$1" == "prim" ];
  then
    RUN_PRIM="yes"
  elif [ "$1" == "bfsmlp" ];
  then
    RUN_BFSMLP="yes"
  elif [ "$1" == "mram" ];
  then
    RUN_MRAM="yes"
  else
    print_help_exit
  fi
else
  print_help_exit
fi

echo "=== Benchmark Configuration ==="
echo "RUN_PRIM: $RUN_PRIM"
echo "RUN_BFSMLP: $RUN_BFSMLP"
echo "RUN_MRAM: $RUN_MRAM"

echo "Writing outputs to: $OUTDIR"
echo ""
mkdir -p $OUTDIR

if [ "$RUN_PRIM" == "yes" ];
then
  echo "=== Running PrIM Benchmarks ==="
  python3 run_prim.py
  mv prim_results.csv $OUTDIR/
  echo "Wrote results to $OUTDIR/prim_results.csv"
  echo ""
fi

if [ "$RUN_BFSMLP" == "yes" ];
then
  echo "=== Running MLP Benchmarks ==="
  python3 run_mlp.py --mode prim --cwd MLP
  mv mlp_results.csv $OUTDIR/
  echo "Wrote results to $OUTDIR/mlp_results.csv"
  echo ""

  echo "=== Running BFS Benchmarks ==="
  python3 run_bfs.py --mode prim --cwd BFS
  mv bfs_results.csv $OUTDIR/
  echo "Wrote results to $OUTDIR/bfs_results.csv"
  echo ""
fi

if [ "$RUN_MRAM" == "yes" ];
then
  echo "=== Running MRAM Benchmark ==="
  
  if [ ! -d ./ime-client-library ];
  then
    echo "Folder ime-client-library missing. Move it here to run the MRAM benchmark."
    exit 1
  fi

  echo "Compiling MRAM Benchmark"
  cmake -DENABLE_UPMEM_BENCHMARK=YES -B build-mram ime-client-library
  cmake --build build-mram --target mram

  echo "Executing MRAM Benchmark"
  ./build-mram/mram > $OUTDIR/mram.csv
fi
