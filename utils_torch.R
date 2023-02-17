
onehot_encoding <- function(data, ind){
  one_hot_data <- data
  if (length(ind) == 1){
    one_hot_data[, ind] <- as.character(one_hot_data[, ind])
    unique_category <- unique(data[, ind])
    no_cat <- length(unique_category)
  }else{
    one_hot_data[, ind] <- lapply(one_hot_data[, ind], as.character)
    unique_category <- lapply(data[, ind], unique)
    no_cat <- unlist(lapply(unique_category, length))
  }
  one_hot <- caret::dummyVars(" ~ .", data = one_hot_data)
  one_hot_data <- data.frame(predict(one_hot, newdata = one_hot_data))
  
  new_ind <- c(ind[1], ind + no_cat)
  cutpoint <- new_ind[-length(new_ind)]
  new_ind <- c(mapply(function(x, y) seq(x, y), 
                      cutpoint, cutpoint + no_cat - 1))
  count <- 1
  for (i in new_ind){
    if (i %in% cutpoint[-1]){
      count <- count + 1
    }
    ind_inactive <- which(one_hot_data[, i] == 0)
    one_hot_data[ind_inactive, i] <- runif(length(ind_inactive), 
                                           min = 0, max = 1/no_cat[count] - 1e-8)
  }
  count <- 1
  for (j in new_ind){
    if (j %in% cutpoint[-1]){
      count <- count + 1
    }
    ind_active <- which(one_hot_data[, j] == 1)
    if (j %in% cutpoint){
      submatrix <- one_hot_data[, j:(j + no_cat[count] - 1)]
    }
    one_hot_data[ind_active, j] <- 2 - rowSums(submatrix)[ind_active]
  }
  return (list(one_hot_data, new_ind))
}

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



uniform_sampling <- function(min = 0, max = 1, nr = 128, nc = 1, matrix = c("z", "h"), hint_rate = NULL, device){
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











