#!/bin/bash

#phenofile='values_FT16.8424.80.del'
#phenofile='values_FT10.8424.dup.del'
#pheno='FT10'
#phenotype file is first argument e.g values_FT16.8424.80.del
#rm test_raw_plink* ; rm train_raw_plink*
echo "$1"
echo "$2"
echo "$3"
#conduct GWAS
#plink2 --glm --mac 20 --bfile completed_big_matrix_binary_new_snps_ids --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --keep name_vector_train.txt --pheno $phenofile
#echo "red"
#cat header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted #formatting
#clump
#plink1.9 --prune --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --clump-kb 250 --clump-p1 1 --clump-p2 1 --clump-r2 0.1 --clump gwas_results_"$1"_in_"$2"_"$3".gsorted --out gwas_results_clumped_"$1"_in_"$2"_"$3"
#head -n 10000 gwas_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > top10ksnps_"$1"_in_"$2"_"$3".txt
#extract top snps
#plink1.9 --prune --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract top10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_"$1"_in_"$2"_"$3"

#plink1.9 --prune --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract top10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_"$1"_in_"$2"_"$3"


#rm name_vector_train.txt ; rm name_vector_test.txt

if [ "$6" == "shuf" ] 
then

#choose shuf set
        pheno=$(cut -f 3 -d ' ' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
	shuf indep_snps_full_dataset_new_snp_ids.prune.in | head -n $5 > shuf_"$5"_snps_"$1"_in_"$2"_"$3".txt

	plink1.9 --pheno $phenofile --prune --bfile completed_big_matrix_binary_new_snps_ids   --keep name_vector_train.txt --extract shuf_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_shuf_"$1"_in_"$2"_"$3"

	plink1.9  --pheno $phenofile --prune --bfile ompleted_big_matrix_binary_new_snps_ids   --keep name_vector_test.txt --extract shuf_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_shuf_"$1"_in_"$2"_"$3"

	mv name_vector_train.txt name_vector_train_"$1"_in_"$2"_"$3".txt ; mv name_vector_test.txt name_vector_test_"$1"_in_"$2"_"$3".txt

fi


if [ "$6" == "mlma" ]
then
        pheno=$(cut -f 3 -d ' ' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
        if [ ! -f train_grm_"$1"_in_"$2"_"$3".grm.bin ]; then
                gcta64 --make-grm  --autosome-num 5 --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --thread-num 32 --out train_grm_"$1"_in_"$2"_"$3"
        fi
        if [ ! -f nested_cv_mlma_out_"$1"_in_"$2"_"$3".mlma ]; then
        gcta64 --mlma --bfile ompleted_big_matrix_binary_new_snps_ids --pheno $phenofile --thread-num 32 --keep name_vector_train.txt --out nested_cv_mlma_out_"$1"_in_"$2"_"$3"
        fi

        cat header2.txt <(sort -g -k 9,9 nested_cv_mlma_out_"$1"_in_"$2"_"$3".mlma | awk '{if ($9 != "-nan") print}' | tail -n +2) > mlma_results_"$1"_in_"$2"_"$3".gsorted
        plink1.9 --keep name_vector_train.txt --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --clump-kb 30 --clump-p1 0.05 --clump-p2 0.1 --clump-r2 0.05 --clump mlma_results_"$1"_in_"$2"_"$3".gsorted --out mlma_results_clumped_"$1"_in_"$2"_"$3"
        head -n $5 mlma_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt
        tmp=$(wc -l mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt | cut -d ' ' -f 1) ; if [ $tmp -lt $5 ] ; then echo "sleeping"; sleep 1d ; fi
        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_mlma_"$1"_in_"$2"_"$3"
        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_mlma_"$1"_in_"$2"_"$3"

fi

if [ "$6" == "mlma20k" ]
then
        pheno=$(cut -f 3 -d ' ' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
        if [ ! -f train_grm_"$1"_in_"$2"_"$3".grm.bin ]; then
                gcta64 --make-grm  --autosome-num 5 --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --thread-num 32 --out train_grm_"$1"_in_"$2"_"$3"
        fi
        if [ ! -f nested_cv_mlma_out_"$1"_in_"$2"_"$3".mlma ]; then
        gcta64 --mlma --bfile completed_big_matrix_binary_new_snps_ids --pheno $phenofile --thread-num 32 --keep name_vector_train.txt --out nested_cv_mlma_out_"$1"_in_"$2"_"$3"
        fi

        cat header2.txt <(sort -g -k 9,9 nested_cv_mlma_out_"$1"_in_"$2"_"$3".mlma | awk '{if ($9 != "-nan") print}' | tail -n +2) > mlma_results_"$1"_in_"$2"_"$3".gsorted
        plink1.9 --keep name_vector_train.txt --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --clump-kb 250 --clump-p1 0.05 --clump-p2 0.2 --clump-r2 0.1 --clump mlma_results_"$1"_in_"$2"_"$3".gsorted --out mlma_results_clumped_"$1"_in_"$2"_"$3"
        head -n $5 mlma_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt
        tmp=$(wc -l mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt | cut -d ' ' -f 1) ; if [ $tmp -lt $5 ] ; then echo "sleeping"; sleep 1d ; fi
        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_mlma_"$1"_in_"$2"_"$3"
        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract mlma_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_mlma_"$1"_in_"$2"_"$3"

fi


if [ "$6" == "top" ]
then
        pheno=$(cut -f 3 -d ' ' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
        #sed "s/'/ /g" name_vector_train.txt | awk '{print $2, $3}' > name_vector_train2.txt; mv name_vector_train2.txt name_vector_train.txt
        #sed "s/'/ /g" name_vector_test.txt | awk '{print $2, $3}' > name_vector_test2.txt; mv name_vector_test2.txt name_vector_test.txt
#choose top set
        plink2 --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --allow-no-sex --glm --mac 20 --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --pheno $phenofile

        if  test -f nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear ; then cat header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted ; fi #formatting
        if test -f nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.logistic ; then cat header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.logistic | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted ; fi #formatting

        #awk '{if ($12 <= 0.01) print}' gwas_results_"$1"_in_"$2"_"$3".gsorted > gwas_results_"$1"_in_"$2"_"$3".gsorted.001
        #echo "WARNING FILTERING RESULTS OF GWAS KESS THAN 0.01"
        #cat header.txt gwas_results_"$1"_in_"$2"_"$3".gsorted.001 > gwas_results_"$1"_in_"$2"_"$3".gsorted.001.filter
        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --clump-kb 250 --clump-p1 0.0005 --clump-p2 0.001 --clump-r2 0.1 --clump gwas_results_"$1"_in_"$2"_"$3".gsorted --out gwas_results_clumped_"$1"_in_"$2"_"$3"

        head -n $5 gwas_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > top_"$5"_snps_"$1"_in_"$2"_"$3".txt

        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_top_"$1"_in_"$2"_"$3"

        plink1.9 --prune --allow-no-sex --pheno $phenofile --bfile completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_top_"$1"_in_"$2"_"$3"


fi


