#!/usr/bin/env bash


set -e
set -x

flake8 logo_classification tests
black logo_classification tests --check
isort logo_classification tests --check-only