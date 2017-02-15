#!/bin/sh

# This is a script to organize PRID2011, generate a list of pairs of tracklets and their corresponding labels.

# download file
download_folder="/home/gabi/Downloads/"
cd $download_folder
wget https://lrs.icg.tugraz.at/datasets/prid/prid_2011.zip
# unzip the file
download_location="/home/gabi/Downloads/prid_2011.zip"
root="/home/gabi/PycharmProjects/uatu/PRID2011"
unzip $download_location -d $root
# delete everything except the 200 first folders of each cam_* folder
cd $root
mv multi_shot/cam_* .
rm -rf multi_shot/ && rm -rf single_shot/ && rm -rf readme.txt
mkdir tmp
cd cam_a
mv `ls | head -n 200` -t ../tmp
rm -rf $root/cam_a/*
cd ../ && mv tmp/* -t cam_a
cd cam_b
mv `ls | head -n 200` -t ../tmp
rm -rf $root/cam_b/*
cd ../ && mv tmp/* -t cam_b
rm -rf tmp
# rename the folders in cam_* from person_nnnn to person_nnnn_a or person_nnnn_b 
cd cam_a
rename 's/(.*)$/$1_a/' person*
cd ../cam_b
rename 's/(.*)$/$1_b/' person*
# move all folders into 1 folder
cd ../
mv cam_*/* .
rm -rf cam*
# generate csv file in the form of: person_0000_a,person_0000_b,1 depending on if they belong to the same subject
cd $root
cd ../

python make_labels_csv.py $root 
