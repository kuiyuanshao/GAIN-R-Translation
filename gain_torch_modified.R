

gain <- function(data, batch_size = 128, hint_rate = 0.9, alpha = 10, n = 5000){
  if (cuda_is_available()){
    device <- torch_device("cuda")
  }else{
    device <- torch_device("cpu")
  }
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  
  norm_result <- normalize(data, Var = 1:nCol)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  norm_data[is.na(norm_data)] <- 0
  
  data_mask <- 1 - is.na(data)
  norm_data <- as.matrix(norm_data)
  
  # Discriminator
  D_W1 <- torch_tensor(xavier_init(c(nCol * 2, nCol), device),
                       requires_grad = T, device = device)
  D_b1 <- torch_tensor(torch_zeros(nCol),
                       requires_grad = T, device = device)
  
  D_W2 <- torch_tensor(xavier_init(c(nCol, nCol), device),
                       requires_grad = T, device = device)
  D_b2 <- torch_tensor(torch_zeros(nCol),
                       requires_grad = T, device = device)
  
  D_W3 <- torch_tensor(xavier_init(c(nCol, nCol), device),
                       requires_grad = T, device = device)
  D_b3 <- torch_tensor(torch_zeros(nCol),
                       requires_grad = T, device = device)
  
  theta_D <- c(D_W1, D_W2, D_W3, D_b1, D_b2, D_b3)
  #Generator
  G_W1 <- torch_tensor(xavier_init(c(nCol * 2, nCol), device),
                       requires_grad = T, device = device)
  G_b1 <- torch_tensor(torch_zeros(nCol),
                       requires_grad = T, device = device)
  
  G_W2 <- torch_tensor(xavier_init(c(nCol, nCol), device),
                       requires_grad = T, device = device)
  G_b2 <- torch_tensor(torch_zeros(nCol),
                       requires_grad = T, device = device)
  
  G_W3 <- torch_tensor(xavier_init(c(nCol, nCol), device),
                       requires_grad = T, device = device)
  G_b3 <- torch_tensor(torch_zeros(nCol),
                       requires_grad = T, device = device)
  
  theta_G <- c(G_W1, G_W2, G_W3, G_b1, G_b2, G_b3)
  
  discriminator <- function(X, H){
    inputs <- torch_cat(tensors = list(X, H), dim = 2)
    D_h1 <- torch_relu(torch_matmul(inputs, D_W1) + D_b1)
    D_h2 <- torch_relu(torch_matmul(D_h1, D_W2) + D_b2)
    D_logit <- torch_matmul(D_h2, D_W3) + D_b3
    D_prob <- torch_sigmoid(D_logit)
    return (D_prob)
  }
  
  generator <- function(X, M){
    inputs <- torch_cat(list(X, M), dim = 2)
    G_h1 <- torch_relu(torch_matmul(inputs, G_W1) + G_b1)
    G_h2 <- torch_relu(torch_matmul(G_h1, G_W2) + G_b2)   
    G_logit <- torch_matmul(G_h2, G_W3) + G_b3
    G_prob <- torch_sigmoid(G_logit)
    return (G_prob)
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
    
    loss <- -torch_mean((1 - M) * torch_log(D_prob + 1e-8))
    mse_train <- torch_mean((M * X - 
                               M * G_sample) ^ 2) / torch_mean(M)
    return (loss + alpha * mse_train)
  }
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  
  D_solver <- optim_adam(theta_D)
  G_solver <- optim_adam(theta_G)
  
  for (i in 1:n){
    
    size <- min(batch_size, nRow)
    idx <- sample_index(nRow, size)
      
    X_mb <- torch_tensor(norm_data[idx,], dtype = torch_float(),
                           device = device)
    M_mb <- torch_tensor(data_mask[idx,], dtype = torch_float(),
                           device = device)
      
    Z_mb <- uniform_sampling(0, 0.01, size, nCol, matrix = "z", 
                             device = device)
    H_mb <- uniform_sampling(0, 1, size, nCol, matrix = "h", 
                               hint_rate, device = device)
      
    H_mb <- M_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * Z_mb
    
    D_solver$zero_grad()
    d_loss <- D_loss(X_mb, M_mb, H_mb)
    d_loss$backward()
    D_solver$step()
    
    G_solver$zero_grad()
    g_loss <- G_loss(X_mb, M_mb, 
                     H_mb, alpha = alpha)
    
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
                              1:nCol)
  imputed_data <- data.frame(as.matrix(imputed_data))
  names(imputed_data) <- names(data)
  
  #imputed_data <- rounding(imputed_data, data)  
  
  return (imputed_data)
}