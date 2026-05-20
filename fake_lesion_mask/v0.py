"""
#!/bin/bash

lesion_mask=${1}
output_mask=${2}
tmp_dir=${3:-".tmp_fakelesion/"}

thr_stable="30"
thr_ero="75"
thr_ero_1="65"
thr_ero_2="73"
thr_new="98"
thr_dil="99"

echo "We start the processing"

# create tmp dir
if [ ! -d "$tmp_dir" ]; then
    mkdir $tmp_dir
fi

# First we label each lesion within the whole mask with unique incremental numbers
python3 -m timelessegv2.tests.connected_comp_skimage ${lesion_mask} "$tmp_dir"lesion_labeled.nii.gz

# count number of lesions
max=`seg_stats "$tmp_dir"lesion_labeled.nii.gz -r | awk '{print $2}'`

# where we gonna store the fake lesions
seg_maths ${lesion_mask} -mul 0 "$tmp_dir"new_lesion_mask.nii.gz

echo "We have a total of $max lesions"

# iterate for each lesion
for num in `seq 1 ${max}` ; do

    # subset mask (only keep lesion $num)
    low=`echo "${num} - 0.5" | bc -l`
    high=`echo "${num} + 0.5" | bc -l`
    seg_maths "$tmp_dir"lesion_labeled.nii.gz -uthr ${high} -thr ${low} -bin "$tmp_dir"single_lesion.nii.gz
    
    random=$(( 1 + RANDOM % 100 )) # this generates a "random" number between 1 and 100

    vol=`seg_stats "$tmp_dir"single_lesion.nii.gz -v` # get volume in mm3
    vol=${vol/.*/}

    # The 50% of the time we keep the lesion equal (stable lesions) source: ferran. He knows how much lesions change wrt volume
    if [ "${random}" -lt "30" ] || [ "${vol}" -gt "2500" ]  ; then
        echo "Lesion $num unchanged - ${vol}"
        seg_maths "$tmp_dir"single_lesion.nii.gz -mul 1 "$tmp_dir"single_lesion.nii.gz
    else
        # Another 25% of the time we reduce the lesion equal (expanding lesions)
        # with different possibilities of expansion/growing
        if [ "${random}" -lt "75" ] ; then
            if [ "${random}" -lt "65" ] ; then
                echo "Lesion $num 1 erosion - ${vol}"
                seg_maths "$tmp_dir"single_lesion.nii.gz -ero 1 "$tmp_dir"single_lesion.nii.gz
            else 
                if [ "${random}" -lt "73" ] || [ "${vol}" -lt "250" ] ; then
                    echo "Lesion $num 2 erosions - ${vol}"
                    seg_maths "$tmp_dir"single_lesion.nii.gz -ero 2 "$tmp_dir"single_lesion.nii.gz
                else
                    echo "Lesion $num 3 erosions - ${vol}"
                    seg_maths "$tmp_dir"single_lesion.nii.gz -ero 3 "$tmp_dir"single_lesion.nii.gz
                fi
            fi
        else
            # 20% of the time the lesion will dissapear (simulating new lesions)
            if [ "${random}" -lt "99" ] ; then
                echo "Lesion $num as new lesion - ${vol}"
                seg_maths "$tmp_dir"single_lesion.nii.gz -mul 0 "$tmp_dir"single_lesion.nii.gz
            else
                # Lesions that are reducing size in future timepoints 3%               
                echo "Lesion $num 1 dilations - ${vol}"
                seg_maths "$tmp_dir"single_lesion.nii.gz -dil 1 "$tmp_dir"single_lesion.nii.gz
            fi
        fi
    fi
    seg_maths "$tmp_dir"new_lesion_mask.nii.gz -add "$tmp_dir"single_lesion.nii.gz -bin "$tmp_dir"new_lesion_mask.nii.gz
done
echo "Done!"

seg_maths "$tmp_dir"new_lesion_mask.nii.gz -range -scl ${output_mask}
rm -r "$tmp_dir"
"""