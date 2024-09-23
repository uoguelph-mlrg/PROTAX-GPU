#!/bin/bash

# This script will find the librhash.1.dylib file and automatically create a symlink named librhash.0.dylib in the same directory. It will also confirm the symlink has been created by listing the contents of the library directory.
# Find the path of the librhash*.dylib file
lib_path=$(find $CONDA_PREFIX/lib/ -name "librhash*.dylib" | grep librhash.1.dylib)

if [ -z "$lib_path" ]; then
    echo "librhash.1.dylib not found."
    exit 1
fi

# Set the path for the new symlink
symlink_path="$CONDA_PREFIX/lib/librhash.0.dylib"

# Create the symbolic link
ln -sf "$lib_path" "$symlink_path"

# Verify the symbolic link was created
ls -l $CONDA_PREFIX/lib | grep librhash

echo "Symbolic link created for librhash.0.dylib"
