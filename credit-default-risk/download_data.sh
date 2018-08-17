#!/usr/bin/env bash

kaggle competitions download home-credit-default-risk -p data
unzip "./data/*.csv.zip" -d "./data"
rm ./data/*.csv.zip
