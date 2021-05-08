#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place logo_classification tests --exclude=__init__.py
black logo_classification tests
isort logo_classification tests