#!/bin/bash

set -e

echo "releasing dionysus"

python -m build
twine check dist/*

cp /app/bin/.pypirc $HOME/.pypirc

twine upload -r pypi dist/*
