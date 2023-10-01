#!/bin/bash

set -e

echo "releasing dl"

python -m build
twine check dist/*

cp /app/bin/.pypirc $HOME/.pypirc

twine upload -r pypi dist/*
