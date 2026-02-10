#!/usr/bin/env bash
set -e
python -m case_study_2.training.train_task_only
python -m case_study_2.training.train_routing_only
python -m case_study_2.training.train_mappo