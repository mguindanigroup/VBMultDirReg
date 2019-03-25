
# generate the data according to: https://github.com/duncanwadsworth/dmbvs

setwd("/dmbvs")
source(file.path("code", "wrapper.R"))
source(file.path("code", "helper_functions.R"))
library(rhdf5)

dir = "/data"
paste(dir,2,".txt",sep="")
set.seed(2019)

for(id in 1:50){
  simdata = simulate_dirichlet_multinomial_regression(n_obs = 100,
                                                      n_vars = 50,
                                                      n_taxa = 50,
                                                      n_relevant_vars = 5,
                                                      n_relevant_taxa = 5,
                                                      beta_min = 0.5,
                                                      beta_max = 1.0,
                                                      signoise = 1.0,
                                                      n_reads_min = 1000,
                                                      n_reads_max = 2000,
                                                      theta0 = 0.01,
                                                      rho = 0.4)
  XX = simdata$XX
  YY = simdata$YY
  design <- cbind(1,XX)
  
  write.table(XX, file =  paste(dir, "cov_duncan_001_id_",id,".txt", sep=""), sep = "\t",
              row.names = TRUE, col.names = NA)
  write.table(YY, file = paste(dir, "response_duncan_001_id_",id,".txt", sep=""), sep = "\t",
              row.names = TRUE, col.names = NA)
  write.table(simdata$betas[,2:51], file = paste(dir, "truebeta_duncan_001_id_",id,".txt", sep=""), sep = "\t",
              row.names = TRUE, col.names = NA)
}