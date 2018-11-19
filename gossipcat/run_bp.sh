#!/bin/bash

echo 'build a machine learning project file system'
mkdir project_
cd project_
touch README.txt
mkdir data notebook code report model log
cd data
mkdir raw train test result report
cd ..
cd code
touch config.json
cd ..
ls -R | grep ":$" | sed -e 's/:$//' \
                        -e 's/[^-][^\/]*\//--/g' \
                        -e 's/^/    /' \
                        -e 's/-/|/'
echo 'done'
cd ..
ls