#Performs Dunnett test and Non-parametric Dunnett Tests (M-robust and MLT-Dunnett)
#From Hothorn 2019 - Robust multiple comparisons against a control group with application in toxicology
#can change alternative hypothesis to onesided in glht
#R-3.6.3/bin/R in venv
#library import order matters
args = commandArgs(trailingOnly=TRUE)
library("broom", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("reshape2", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("robustbase", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("mvtnorm", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("TH.data", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("multcomp", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("nparcomp", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("variables", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("basefun", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("mlt", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")
library("ggplot2", lib="/hpc/hers_en/rmclaughlin/r_libs/x86_64-pc-linux-gnu-library/3.6/")

#e.g  grep 'is \[' cv_grid_all_ml_312_mlma_1k_ft16.txt  | grep -v 'Variance' | grep -v 'AUC' | cut -f '2' -d '['   | cut -d ']' -f 1 #remove baseline and add gblup
sprintf(args[1])
results_table <- scan(args[1], sep=',') #just the scores separated by commas, each model on a different line with no rownames!
results_table <- matrix(results_table, ncol=4, byrow=TRUE)
rownames(results_table) <- c("gBLUP", "SVM", "RBF", "LASSO", "Ridge", "RF", "FNN", "CNN")
results_table[results_table < 0] <- 0 # convert negatiive values to 0
results_table <- melt(results_table)

#value is the results values e.g r2
#Var1 is the model names

model_r2 <- lm(value~Var1, data=results_table) #linear model
rob_model_r2 <- lmrob(value~Var1, data=results_table, setting="KS2014") # robust M estimators #see ref 13 from paper
dunnett_result <- summary(glht(model_r2,linfct=mcp(Var1="Dunnett"))) #parametric Dunnett Test
print("DUNNETT TEST")
dunnett_result # #print results of dunnett test
dunnett <- fortify(dunnett_result)

print("M-robust DUNETT")
rob_dunnett_result <- summary(glht(rob_model_r2,linfct=mcp(Var1="Dunnett"))) #M-robust Dunnett 
rob_dunnett_result
rob_dunnett <- fortify(rob_dunnett_result)

print("MLT-DUNNETT")
yvar <- numeric_var("value", support=quantile(results_table$value,prob=c(.01,.99))) #MLT
bstorder <- 5 #order of Bernstein polynomical # recommednded between 5 and 10
yb <- Bernstein_basis(yvar,ui="increasing",order=bstorder) # Bernstein polynominal
ma <-ctm(yb, shifting = ~ Var1, todistr="Normal", data=results_table) # condit transf mod
m_mlt <- mlt(ma, data=results_table) # most likely transformation
K <- diag(length(coef(m_mlt))) # contrast matrix
rownames(K) <- names(coef(m_mlt))
matr <- bstorder+1
K <- K[-(1:matr),] #for order 5 Bernstein
C <- glht(m_mlt, linfct= K) # MLT-Dunnett test
summary(C) # print results of MLT-Dunnett
CMLT <- fortify(summary(C))

print("SMALL SAMPLE MLT-DUNNETT")
tC <- glht(m_mlt, linfct= K, df=model_r2$.df.residual) # MLT for small sample size (t-distribution)
summary(tC)
tCMLT <- fortify(summary(tC))

num_models <- nrow(K)-1
pr2 <- cbind(dunnett[,c(1,num_models)],rob_dunnett[,num_models],CMLT[,num_models], tCMLT[,num_models])
pr2
