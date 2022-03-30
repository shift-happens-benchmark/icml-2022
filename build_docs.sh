#!/bin/bash

DOCKERNAME=shifthappens/sphinx
docker build -t $DOCKERNAME -f - docs << EOF
from sphinxdoc/sphinx
add requirements.txt requirements.txt
run pip install --no-cache -r requirements.txt
EOF

docker run --rm --user $(id -u) -v $(pwd)/docs:/docs -v $(pwd):/code -e PYTHONPATH=/code $DOCKERNAME make html
