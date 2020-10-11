#!/usr/bin/env bash

N_TRAIN_GAMES=10
N_TEST_GAMES=10

OUTPUT_DIR="../../games/cleanup"
EASY_DIR="$OUTPUT_DIR/easy"
MEDIUM_DIR="$OUTPUT_DIR/medium"
DIFFICULT_DIR="$OUTPUT_DIR/hard"

mkdir ${OUTPUT_DIR}
mkdir ${EASY_DIR}
mkdir ${MEDIUM_DIR}
mkdir ${DIFFICULT_DIR}

seed=1234

function make_games {
    n=$1
    level=$2
    path=$3
    split=$4
    for (( i=1; i<=$n; i++ ))
    do
        seed=$(($seed + 1))
        python iqa_cleanup.py --level ${level} --seed ${seed} --output_dir ${path} ${split}
    done
}

for l in 1 2 3
do
    case ${l} in
     1)
          dir=${EASY_DIR}
          ;;
     2)
          dir=${MEDIUM_DIR}
          ;;
     3)
          dir=${DIFFICULT_DIR}
          ;;
    esac
    make_games ${N_TRAIN_GAMES} ${l} "$dir/train" --train
    make_games ${N_TEST_GAMES} ${l} "$dir/valid" --train
    make_games ${N_TEST_GAMES} ${l} "$dir/test" --test
done




