#!/bin/bash

# run GAP dataset

counter=0
exp="exp"

for i in 1 30 50
do
    for j in 1 30 50
	do
	    for k in 1 5 50
		do
		    for l in 256
			do
			    for m in 3 5 7 10
				do
                    counter=$[counter+1]
                    echo "Running experiment $counter -w_adv $i -w_con $j -w_lat $k -nz $l -extralayers $m "
                    CUDA_VISIBLE_DEVICES=1 python train.py --dataset GAP --isize 64 --niter 10 --w_adv $i --w_con $j --w_lat $k --nz $l --extralayers $m --dataroot /data/tilonsm/Documents/RoadSurfaceDamageUAV/data/processed/f-anogan3 --nc 1 --ngpu 1 --outf /data/tilonsm/Documents/RoadSurfaceDamageUAV/src/models/skip-ganomaly/output/run_GAP --name $exp$counter --print_freq 50 --save_image_freq 100 --save_test_images
				done
			done
		done
	done
done

exit 0
