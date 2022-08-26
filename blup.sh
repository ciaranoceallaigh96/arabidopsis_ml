#Uses SNP set from nested_grid search (top or shuf)
#estimate random effects in a MLM by BLUP
#computes blup solutions for snps
#Note --autosome-num doesnt seem to make a difference
pheno=$1 #eg FT10 or FT16
snps=$2 #top or shuf
NOW=$( date '+%F_%H:%M:%S' )
set_size=$3 #e.g 5000 10000
echo $NOW >> "$pheno"_gblup_train_only_grm_cv_test.prscores ; \
echo "WARNING: DO NOT RUN THIS SCRIPT MORE THAN ONCE AT A TIME. THE ID FILES WILL BE OVERWRITTEN BY THE OTHER SCRIPT." >> "$pheno"_gblup_train_only_grm_cv_test.prscores ; \
for i in {1..4} ; do \
cut -d ' ' -f 1-2 train_raw_plink_"$snps"_"$i"_in_4_out.raw > arabadopsis/phenotypes/train_ids.txt ; \
cut -d ' ' -f 1-2 test_raw_plink_"$snps"_"$i"_in_4_out.raw > arabadopsis/phenotypes/test_ids.txt \
; \
GCTA_manual_install/gcta64 \
--bfile completed_big_matrix_binary_new_snps_ids \
--make-grm \
--autosome-num 5 \
--thread-num 32 \
--out ./full_completed_big_matrix_binary_grm_"$pheno"_train_cv_"$i" \
--keep ./arabadopsis/phenotypes/train_ids.txt \
; \
GCTA_manual_install/gcta64 \
--reml \
--grm ./full_completed_big_matrix_binary_grm_"$pheno"_train_cv_"$i" \
--pheno .arabadopsis/phenotypes/values_"$pheno".txt \
--reml-pred-rand \
--out ./full_"$pheno"_blup_solutions_train_cv_"$i" \
--keep ./arabadopsis/phenotypes/train_ids.txt \
; \
GCTA_manual_install/gcta64 \
--bfile ./completed_big_matrix_binary_new_snps_ids \
--blup-snp ./full_"$pheno"_blup_solutions_train_cv_"$i".indi.blp \
--extract "$snps"_"$set_size"_snps_"$i"_in_4_out.txt \
--keep arabadopsis/phenotypes/train_ids.txt \
--out ./full_"$pheno"_gblup_snp_FX_train_only_grm_cv_"$i" \
; \
plink1.9 \
--bfile completed_big_matrix_binary_new_snps_ids \
--keep arabadopsis/phenotypes/test_ids.txt \
--out ./full_"$pheno"_gblup_train_only_grm_cv_"$i"_test \
--pheno arabadopsis/phenotypes/values_"$pheno".txt \
--score ./full_"$pheno"_gblup_snp_FX_train_only_grm_cv_"$i".snp.blp 1 2 3 \
; \
python r2_score.py ./full_"$pheno"_gblup_train_only_grm_cv_"$i"_test.profile "$i" >> "$pheno"_gblup_train_only_grm_cv_test.prscores \
; done
