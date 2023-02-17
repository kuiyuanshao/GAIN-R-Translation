library(caret)
library(torch)
library(progress)

gain <- function(data, vartype, batch_size = 128, hint_rate = 0.9, alpha = 10, n = 5000){
  if (cuda_is_available()){
    device <- torch_device("cuda")
  }else{
    device <- torch_device("cpu")
  }
  ind <- which(vartype == 0)
  if (length(ind) > 0){
    result <- onehot_encoding(data, ind)
    one_hot_data <- result[[1]]
    new_ind <- result[[2]]
    num_cols <- (1:dim(one_hot_data)[2])[-new_ind]
  }else{
    one_hot_data <- data
    num_cols <- 1:dim(one_hot_data)[2]
  }
  
  nRow <- dim(one_hot_data)[1]
  nCol <- dim(one_hot_data)[2]
  
  H_dim <- nCol
  
  norm_result <- normalize(one_hot_data, Var = num_cols)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  norm_data[is.na(norm_data)] <- 0
  
  data_mask <- 1 - is.na(one_hot_data)
  norm_data <- as.matrix(norm_data)
  
  # Discriminator
  
  G_layer <- nn_sequential(nn_linear(nCol * 2, H_dim),
                           nn_relu(),
                           nn_linear(H_dim, H_dim),
                           nn_relu(),
                           nn_linear(H_dim, nCol),
                           nn_relu(),
                           nn_linear(H_dim, H_dim),
                           nn_sigmoid())$to(device = device)
  D_layer <- nn_sequential(nn_linear(nCol * 2, H_dim),
                           nn_relu(),
                           nn_linear(H_dim, H_dim),
                           nn_relu(),
                           nn_linear(H_dim, nCol),
                           nn_sigmoid())$to(device = device)
  generator <- function(X, M){
    input <- torch_cat(list(X, M), dim = 2)
    return (G_layer(input))
  }
  discriminator <- function(X, H){
    input <- torch_cat(list(X, H), dim = 2)
    return (D_layer(input))
  }
  
  G_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    G_loss1 <- -torch_mean((1 - M) * torch_log(D_prob + 1e-8))
    mse_loss <- torch_mean((M * X - M * G_sample) ^ 2) / torch_mean(M)
    
    return (G_loss1 + alpha * mse_loss)
  }
  D_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    D_loss1 <- -torch_mean(M * torch_log(D_prob + 1e-8) + (1 - M) * torch_log(1 - D_prob + 1e-8))
    return (D_loss1)
  }
  
  G_solver <- optim_adam(G_layer$parameters)
  D_solver <- optim_adam(D_layer$parameters)
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  sample_mat <- function(){
    size <- min(batch_size, nRow)
    idx <- sample_index(nRow, size)
    
    X_mb <- torch_tensor(norm_data[idx,], dtype = torch_float(),
                         device = device)
    M_mb <- torch_tensor(data_mask[idx,], dtype = torch_float(),
                         device = device)
    
    Z_mb <- uniform_sampling(0, 0.01, batch_size, nCol, matrix = "z", 
                             device = device)
    H_mb <- uniform_sampling(0, 1, batch_size, nCol, matrix = "h", 
                             1 - hint_rate, device = device)
    
    H_mb <- M_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * Z_mb
    return (list(X_mb, H_mb, M_mb))
  }
  
  
  for (i in 1:n){
    D_solver$zero_grad()
    samp_result <- sample_mat()
    X_mb <- samp_result[[1]]
    H_mb <- samp_result[[2]]
    M_mb <- samp_result[[3]]
    d_loss <- D_loss(X_mb, M_mb, H_mb)
    d_loss$backward()
    D_solver$step()
    
    G_solver$zero_grad()
    samp_result <- sample_mat()
    X_mb <- samp_result[[1]]
    H_mb <- samp_result[[2]]
    M_mb <- samp_result[[3]]
    g_loss <- G_loss(X_mb, M_mb, H_mb)
    
    g_loss$backward()
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   "))
    Sys.sleep(2 / 100)
  }
  
  Z_mb <- uniform_sampling(0, 0.01, nRow, nCol, matrix = "z", device = device) 
  M_mb <- torch_tensor(data_mask, dtype = torch_float(),
                       device = device)
  X_mb <- torch_tensor(norm_data, dtype = torch_float(),
                       device = device)
  X_mb <- M_mb * X_mb + (1 - M_mb) * Z_mb 
  
  imputed_data <- generator(X_mb, M_mb)
  imputed_data <- M_mb * X_mb + (1 - M_mb) * imputed_data
  imputed_data <- imputed_data$detach()
  imputed_data <- renormalize(imputed_data, norm_parameters, 
                              num_cols)
  imputed_data <- data.frame(as.matrix(imputed_data))
  names(imputed_data) <- names(one_hot_data)
  newcol <- dim(imputed_data)[2]:(dim(imputed_data)[2] + length(ind))
  imputed_data$Z <- ifelse(imputed_data$Z1 > imputed_data$Z0, 1, 0)
  #imputed_data <- rounding(imputed_data, data)  
  
  return (imputed_data)
}