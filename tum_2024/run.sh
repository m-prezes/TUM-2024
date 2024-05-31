#!/bin/bash

# Sprawdź, czy poetry jest zainstalowane
if ! command -v poetry &> /dev/null
then
    echo "Poetry could not be found. Please install poetry first."
    exit 1
fi

# Uruchom script_cifar.py
echo "Running script_cifar.py..."
poetry run python script_cifar.py
if [ $? -ne 0 ]; then
    echo "script_cifar.py failed to run."
    exit 1
fi

# Uruchom script_mnist.py
echo "Running script_mnist.py..."
poetry run python script_mnist.py
if [ $? -ne 0 ]; then
    echo "script_mnist.py failed to run."
    exit 1
fi


echo "Both scripts ran successfully."
