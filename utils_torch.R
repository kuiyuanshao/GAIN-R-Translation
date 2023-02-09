


normalize <- function(data, parameters = NULL, Var){
  norm_data <- data
  
  if (is.null(parameters)){
    min_val <- max_val <- rep(0, length(Var))
    for (i in Var){
      min_val[i] <- min(norm_data[, i], na.rm = T)
      norm_data[, i] <- norm_data[, i] - min(norm_data[, i], na.rm = T)
      max_val[i] <- max(norm_data[, i], na.rm = T)
      norm_data[, i] <- norm_data[, i] / (max(norm_data[, i], na.rm = T) + 1e-6)
    }
    norm_parameters <- list(min_val = min_val,
                            max_val = max_val)
  }else{
    min_val <- parameters$min_val
    max_val <- parameters$max_val
    
    for (i in Var){
      norm_data[, i] <- norm_data[, i] - min_val[i]
      norm_data[, i] <- norm_data[, i] / (max_val[i] + 1e-6)
    }
    norm_parameters <- parameters
  }
  return (list(norm_data = norm_data, norm_parameters = norm_parameters))
}

renormalize <- function(norm_data, norm_parameters, Var){
  
  min_val <- norm_parameters$min_val
  max_val <- norm_parameters$max_val
  
  renorm_data <- norm_data
  for (i in Var){
    renorm_data[, i] <- renorm_data[, i] * (max_val[i] + 1e-6)
    renorm_data[, i] <- renorm_data[, i] + min_val[i]
  }
  
  return (renorm_data)
}



uniform_sampling <- function(min, max, nr, nc, matrix = c("z", "h"), hint_rate = NULL, device){
  random_unif <- runif(min = min, max = max, n = nr * nc)
  unif_matrix <- matrix(random_unif, nrow = nr, ncol = nc, byrow = T)
  if (matrix == "z"){
    return (torch_tensor(unif_matrix, device = device)) 
  }else{
    H <- 1 * (unif_matrix > hint_rate)
    return(torch_tensor(H, device = device))
  }
}

rounding <- function(imputed_data, data){
  num <- unlist(lapply(data, is.numeric))
  for (i in num){
    
    nchar(strsplit(as.character(), "\\.")[[1]][2])
  }
}



sample_index <- function(total, batch_size){
  total_idx <- sample(total)
  batch_idx <- total_idx[1:batch_size]
  return (batch_idx)

}


xavier_init <- function(size, device){
  xavier_stddev <- 1 / sqrt(size[1] / 2)
  return (torch_randn(size, device = device) * xavier_stddev)
}











