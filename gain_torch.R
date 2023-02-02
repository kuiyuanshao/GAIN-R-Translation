

gain <- function(data, batch_size = 128, hint_rate = 0.9, alpha = 10, n = 5000){
  if (cuda_is_available()){
    device <- torch_device("cuda")
  }else if (backends_mps_is_available()){
    device <- torch_device("mps")
  }else{
    device <- torch_device("cpu")
  }
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  
  norm_result <- normalize(data)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  norm_data[is.na(norm_data)] <- 0
  
  
  data_mask <- 1 - is.na(data)
  norm_data <- as.matrix(norm_data)
  #data_mask <- torch_tensor(data_mask, dtype = torch_float(), device = device)
  #norm_data <- torch_tensor(as.matrix(norm_data), dtype = torch_float(), device = device)
  
  layer <- nn_sequential(
    nn_linear(nCol * 2, nCol),
    nn_relu(),
    nn_linear(nCol, nCol),
    nn_relu(),
    nn_linear(nCol, nCol),
    nn_sigmoid()
  )$to(device = device)
  
  D_solver <- optim_adam(layer$parameters)
  G_solver <- optim_adam(layer$parameters)
  
  discriminator <- function(X, H){
    inputs <- torch_cat(list(X, H), dim = 2)
    return (layer(inputs))
  }
  
  generator <- function(X, M){
    inputs <- torch_cat(list(X, M), dim = 2)
    return (layer(inputs))
  }
  
  D_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    loss <- -torch_mean(M * torch_log(D_prob + 1e-8) + (1 - M) * 
                          torch_log(1 - D_prob + 1e-8))
    return (loss)
  }
  
  G_loss <- function(X, M, H, alpha){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    G_loss <- -torch_mean((1 - M) * torch_log(D_prob + 1e-8))
    MSE_loss <- torch_mean((M * X - M * G_sample) ^ 2) / torch_mean(M)
    
    return (G_loss + alpha * MSE_loss)
  }
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  
  for (i in 1:n){
    D_solver$zero_grad()
    
    sampling <- function(){
      size <- min(batch_size, nRow)
      idx <- sample_index(nRow, size)
      
      x_mb <- torch_tensor(norm_data[idx,], dtype = torch_float(),
                           device = device)
      m_mb <- torch_tensor(data_mask[idx,], dtype = torch_float(),
                           device = device)
      
      z_mb <- uniform_sampling(0, 0.01, size, nCol, matrix = "z")
      h_mb <- uniform_sampling(0, 1, size, nCol, matrix = "h", 
                               1 - hint_rate)
      
      h_mb <- m_mb * h_mb
      x_mb <- m_mb * x_mb + (1 - m_mb) * z_mb
      return (list(x_mb = x_mb, h_mb = h_mb, m_mb = m_mb))
    }
    samp_result <- sampling()
    d_loss <- D_loss(samp_result$x_mb, samp_result$m_mb, 
                     samp_result$h_mb)
    d_loss$backward()
    D_solver$step()
    
    G_solver$zero_grad()
    samp_result <- sampling()
    g_loss <- G_loss(samp_result$x_mb, samp_result$m_mb, 
                     samp_result$h_mb, alpha = alpha)
    g_loss$backward()
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   "))
    Sys.sleep(2 / 100)
  }
  
  Z_mb <- uniform_sampling(0, 0.01, nRow, nCol, matrix = "z") 
  M_mb <- torch_tensor(data_mask, dtype = torch_float(),
                       device = device)
  X_mb <- torch_tensor(norm_data, dtype = torch_float(),
                       device = device)
  X_mb <- M_mb * X_mb + (1-M_mb) * Z_mb 
  
  imputed_data <- generator(X_mb, M_mb)
  imputed_data <- M_mb * X_mb + (1 - M_mb) * imputed_data
  
  imputed_data <- renormalize(imputed_data, norm_parameters)
  imputed_data <- as.data.frame(as.matrix(imputed_data))
  names(imputed_data) <- names(data)
  imputed_data <- rounding(imputed_data, data)  
  
  return (imputed_data)
}