#!/bin/sh

CNN_MODEL=$2
OUTPUT_DIR=$3
NET_NAME=$4


if [ $1 = kr260 ]; then
            ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
            TARGET=kr260
            echo "-----------------------------------------"
            echo "COMPILING MODEL FOR KR260..."
            echo "-----------------------------------------"

else
      echo  "Target not found. Valid choices is kr260 . exiting"
      exit 1
fi


compile() {
    vai_c_tensorflow2 \
	--model           ./${CNN_MODEL} \
	--arch            ${ARCH} \
	--output_dir      ${OUTPUT_DIR} \
	--net_name        ${NET_NAME} 

}


compile 


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
