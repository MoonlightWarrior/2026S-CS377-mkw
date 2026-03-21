#!/bin/bash
# Doesn't work with Mac OS 26.x and newer as AGL is removed

DOLPHIN_URL="https://github.com/Felk/dolphin.git"
DOLPHIN_DIR="dolphin"

if [ -d "$DOLPHIN_DIR" ]; then
    echo "Dolphin already cloned."
else
    echo "Cloning Dolphin..."
    git clone --depth=1 $DOLPHIN_URL $DOLPHIN_DIR
    cd $DOLPHIN_DIR
    git submodule update --init --recursive --progress --depth=1
    echo "Dolphin cloned successfully."
    cd ..
fi

# Entering the source code
cd $DOLPHIN_DIR

# Compiling dolphin
mkdir -p Build
cd Build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSKIP_POSTPROCESS_BUNDLE=ON                           
make -j $(nproc)

# Deleting the old dolphin we had (if there is one)
rm -rf ../../dolphin0

# Copying the compiled dolphin
mkdir -p ../../dolphin0
cp -r Binaries/DolphinQt.app ../../dolphin0

# Making dolphin portable
touch ../../dolphin0/portable.txt

cd ../..

# Deleting the source code
rm -rf $DOLPHIN_DIR
