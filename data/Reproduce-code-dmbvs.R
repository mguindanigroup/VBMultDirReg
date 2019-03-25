###### the code is dedicated to reproduce duncan's results at:
######https://github.com/duncanwadsworth/dmbvs


setwd("/dmbvs")
source(file.path("code", "wrapper.R"))
source(file.path("code", "helper_functions.R"))
library(rhdf5)


### start experiment

run_exp <- function(){
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
  start.time <- Sys.time()
  results = dmbvs(XX = simdata$XX[,-1], YY = simdata$YY, 
                  intercept_variance = 10, slab_variance = 10, 
                  bb_alpha = 0.02, bb_beta = 1.98, GG = 500L, thin = 10L, burn = 100L,
                  exec = file.path(".", "code", "dmbvs.x"), output_location = ".")
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(time.taken)
  return(results)
}
results <- run_exp()


beta_accept <- results$beta
reshape_pred_beta <- matrix(colMeans(beta_accept) , nrow=40, ncol=100,byrow = T)
# beta_mc_acc_ratio <- matrix( results$beta_accept, nrow=40, ncol=100,byrow = T)

find_non_zero_beta <- function(beta_matrix){
ind <- 1
l <- list()
for (i in 1:n_taxa){
  for (j in 1:n_vars){
    if (abs(beta_matrix[i,j])>0.3){
      l[[ind]] <- c(i,j)
      ind <- ind+1
    }
  }
}
return(l)
}

################################  repetition experiment: dmbvs
arr_est_beta_duncan = array(0, dim=c(50,2500))
arr_acc_ratio_duncan = array(0, dim=c(10,2500))
set.seed(2019)
for(id in 1:10){
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
  print('finish generate data and start to fit model')
  XX = simdata$XX[,2:51] # cov matrix shape=(n,p)
  # print(dim(XX))
  YY = simdata$YY # response matrix, shape=(n,q)
  results = dmbvs(XX = XX, YY = YY, 
                intercept_variance = 10, slab_variance = 10, 
                bb_alpha = 0.02, bb_beta = 1.98, GG = 1500L, thin = 10L, burn = 500L,
                exec = file.path(".", "code", "dmbvs.x"), output_location = ".")
  print('MCMC finished!')
  beta_accept <- results$beta
  beta_accept[abs(beta_accept)>0]=1
  beta_accept <- colMeans(beta_accept)
  
  stopifnot(max(beta_accept)==1)
  arr_acc_ratio_duncan[id,] <- beta_accept
  # mean.beta <- abs(colMeans(beta_accept))
  # arr_est_beta_duncan[id,] <- mean.beta
}

### save the results of dmbvs
# results_dir <- "/Users/luyadong/Documents/git_project/dmbvs/rep_50_results/"
# results_filename <- paste(results_dir, "est_beta_duncan_001.h5", sep="")
# h5createFile(file =  results_filename)
# h5write(arr_est_beta_duncan, results_filename, name="est_beta_duncan_001")


results_dir <- "/Users/luyadong/Documents/git_project/dmbvs/rep_50_results/"
results_filename <- "acc_beta_duncan_001.h5"

writeresults <- function(results_dir,results_filename, results){
  results_filename_dir <- paste(results_dir, results_filename, sep="")
  print(results_filename_dir)
  h5createFile(file =  results_filename_dir)
  h5write(results, results_filename_dir, name="est_beta_acc_ratio")
}
writeresults(results_dir,results_filename, arr_acc_ratio_duncan)

### example on matrix flatten
matrix(seq(20), nrow = 4, ncol = 5, byrow = T)
as.vector(matrix(seq(20), nrow = 4, ncol = 5, byrow = T)) #
matrix(as.vector(matrix(seq(20), nrow = 4, ncol = 5, byrow = T)), nrow = 4, ncol = 5, byrow = F)



results = dmbvs(XX = X, YY = Y, 
                intercept_variance = 10, slab_variance = 10, 
                bb_alpha = 0.02, bb_beta = 1.98, GG = 1500L, thin = 10L, burn = 500L,
                exec = file.path(".", "code", "dmbvs.x"), output_location = ".")
beta_accept <- results$beta
beta_accept[abs(beta_accept)>0]=1
beta_accept <- colMeans(beta_accept)
write.table(beta_accept, file = "duncan_real_data.txt", sep = "\t",
            row.names = TRUE, col.names = NA)
