#!/usr/bin/env bash

kaggle competitions download home-credit-default-risk -p data

if [ $? -ne 0 ]
then
    exit
fi

unzip "./data/*.csv.zip" -d "./data"

if [ $? -ne 0 ]
then
    exit
fi

rm ./data/*.csv.zip
