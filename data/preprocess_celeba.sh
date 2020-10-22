#! /bin/sh

cd data
unzip img_align_celeba.zip
mkdir CelebA_trainval
mv img_align_celeba CelebA_trainval
mkdir -p CelebA_test/img_align_celeba

trainval_dir="CelebA_trainval/img_align_celeba"
test_dir="CelebA_test/img_align_celeba"
while IFS=' ' read -r fpath status; do
    if [ $status -eq 2 ]; then
        mv $trainval_dir"/"$fpath $test_dir"/"$fpath
    fi
done < list_eval_partition.txt
