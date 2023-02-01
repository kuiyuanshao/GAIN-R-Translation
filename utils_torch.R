


normalize <- function(data, parameters = NULL){
  nVar <- dim(data)[2]
  norm_data <- data
  
  if (is.null(parameters)){
    min_val <- max_val <- rep(0, nVar)
    for (i in 1:nVar){
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
    
    for (i in 1:nVar){
      norm_data[, i] <- norm_data[, i] - min_val[i]
      norm_data[, i] <- norm_data[, i] / (max_val[i] + 1e-6)
    }
    norm_parameters <- parameters
  }
  return (list(norm_data = norm_data, norm_parameters = norm_parameters))
}

renormalize <- function(norm_data, norm_parameters){
  
  min_val <- norm_parameters$min_val
  max_val <- norm_parameters$max_val
  
  nVar <- dim(norm_data)[2]
  renorm_data <- norm_data
  for (i in 1:nVar){
    renorm_data[, i] <- renorm_data[, i] * (max_val[i] + 1e-6)
    renorm_data[, i] <- renorm_data[, i] + min_val[i]
  }
  
  return (renorm_data)
}

rounding <- function(imputed_data, data_x){
  
  nVar <- dim(data_x)[2]
  rounded_data <- imputed_data
  
  for (i in 1:nVar){
    temp <- data_x[!is.na(data_x[, i]), i]
    
    if (length(unique(temp)) < 20){
      rounded_data[, i] <- round(rounded_data[, i])
    }
  }
  
  return (rounded_data)
}


uniform_sampling <- function(min, max, nr, nc, matrix = c("z", "h"), hint_rate = NULL){
  random_unif <- runif(min = min, max = max, n = nr * nc)
  unif_matrix <- matrix(random_unif, nrow = nr, ncol = nc, byrow = T)
  if (matrix == "z"){
    return (torch_tensor(unif_matrix, device = device)) 
  }else{
    H <- unif_matrix > hint_rate
    H <- 1 * H
    return(torch_tensor(H, device = device))
  }
}



sample_index <- function(total, batch_size){
  total_idx <- sample(total)
  batch_idx <- total_idx[1:batch_size]
  return (batch_idx)

}














