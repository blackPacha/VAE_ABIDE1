#!/bin/bash

# example:
# ./save_in_array.sh model_number outdirname

save_in_array_npy() {
	python vae_save_in_array.py -dirname ${1}/output_${2} -n ${2} -outdir ${1}
}

save_in_array_npy ${1} ${2}
 
