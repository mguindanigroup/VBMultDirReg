
# generate the data according to: https://github.com/duncanwadsworth/dmbvs

setwd("/dmbvs")
source(file.path("code", "wrapper.R"))
source(file.path("code", "helper_functions.R"))
library(rhdf5)

dir = "/data"
paste(dir,2,".txt",sep="")
set.seed(20192)
for(id in 1:200){
  print(paste0('start generating dataset',id))
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
                                                      theta0 = 0.1,
                                                      rho = 0.4)
  XX = simdata$XX # design matrix shape=(n,p+1)
  YY = simdata$YY # response matrix, shape=(n,q)
  print(dim(YY))
}
filename <- paste(dir, "rep_50_001.h5", sep="")
h5createFile(file =  filename)
h5write(arr_XX, filename, name="cov_duncan_001")
h5write(arr_YY, filename, name="response_duncan_001")
h5write(arr_true_beta, filename, name="truebeta_duncan_001")