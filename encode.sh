#!/bin/bash

#Launch VAE encoding:
#- precise output directory  
#example:
#./encode.sh model_number outdirname


actualize_checkpoint() {
	sudo sed -i 0,/${1}/s//${2}/ ./models/model_autoencoder_demo/models/checkpoint
}

encode_vae() {
	mkdir -p ${2}/output_${1}

	net_autoencoder inference -c my_collection_network/vae_config.ini \
        	--inference_type encode \
        	--save_seg_dir ${2}/output_${1} \
        	--inference_iter -1 --name my_collection_network.my_vae.VAE
}

# Guessing max_iter=15000 in vae_config.ini
actualize_checkpoint 15000 ${1}
encode_vae ${2} ${1}

./save_in_array.sh ${1} ${2}
