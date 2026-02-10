#!/usr/bin/env bash
set -e
python -m case_study_2.evaluation.eval_runner --method static --seeds 1,2,3,4,5
python -m case_study_2.evaluation.eval_runner --method greedy --seeds 1,2,3,4,5