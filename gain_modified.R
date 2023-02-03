


gain_imp <- function(data, batch_size = 128, hint_rate = 0.9, 
                     alpha = 10, n = 5000){
  data_r <- data$R
  data <- data[, -((dim(data)[2] - 1):dim(data)[2])]
  data_m <- 1 - is.na(data)
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  
  #nCol <- nCol
  
  norm_result <- normalize(data)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  norm_data[is.na(norm_data)] <- 0
  
  X <- tf$compat$v1$placeholder(tf$float32, shape = list(NULL, nCol))
  R <- tf$compat$v1$placeholder(tf$float32, shape = list(NULL, 
                                                         as.integer(1)))
  M <- tf$compat$v1$placeholder(tf$float32, shape = list(NULL, nCol))
  H <- tf$compat$v1$placeholder(tf$float32, shape = list(NULL, 
                                                         as.integer(1)))
  
  
  D_W1 <- tf$Variable(xavier_init(as.integer(c(nCol + 1, nCol))))
  D_b1 <- tf$Variable(tf$zeros(shape = as.integer(nCol)))
  
  D_W2 <- tf$Variable(xavier_init(as.integer(c(nCol, nCol))))
  D_b2 <- tf$Variable(tf$zeros(shape = as.integer(nCol)))
  
  D_W3 <- tf$Variable(xavier_init(as.integer(c(nCol, nCol))))
  D_b3 <- tf$Variable(tf$zeros(shape = c(nCol)))
  
  
  theta_D <- c(D_W1, D_W2, D_W3, D_b1, D_b2, D_b3)
  
  G_W1 <- tf$Variable(xavier_init(as.integer(c(nCol + 1, nCol)))) 
  G_b1 <- tf$Variable(tf$zeros(shape = as.integer(nCol)))
  
  G_W2 <- tf$Variable(xavier_init(as.integer(c(nCol, nCol))))
  G_b2 <- tf$Variable(tf$zeros(shape = c(nCol)))
  
  G_W3 <- tf$Variable(xavier_init(as.integer(c(nCol, nCol))))
  G_b3 <- tf$Variable(tf$zeros(shape = c(nCol)))
  
  theta_G <- c(G_W1, G_W2, G_W3, G_b1, G_b2, G_b3)
  
  generator <-  function(x, r){
    # Concatenate Mask and Data
    inputs <- tf$concat(values = c(x, r), axis = as.integer(1)) 
    G_h1 <- tf$nn$relu(tf$matmul(inputs, G_W1) + G_b1)
    G_h2 <- tf$nn$relu(tf$matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob <- tf$nn$sigmoid(tf$matmul(G_h2, G_W3) + G_b3) 
    return (G_prob)
  }
  # Discriminator
  discriminator <- function(x, h){
    # Concatenate Data and Hint
    inputs <- tf$concat(values = c(x, h), axis = as.integer(1))
    D_h1 <- tf$nn$relu(tf$matmul(inputs, D_W1) + D_b1)  
    D_h2 <- tf$nn$relu(tf$matmul(D_h1, D_W2) + D_b2)
    D_logit <- tf$matmul(D_h2, D_W3) + D_b3
    D_prob <- tf$nn$sigmoid(D_logit)
    return (D_prob)
    
  }
  
  G_sample <- generator(X, R)
  
  # Combine with observed data
  Hat_X <- X * M + G_sample * (1 - M)
  
  # Discriminator
  D_prob <- discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp <- -tf$reduce_mean(M * tf$math$log(D_prob + 1e-8) + 
                                   (1-M) * tf$math$log(1 - D_prob + 1e-8)) 
  
  G_loss_temp <- -tf$reduce_mean((1-M) * tf$math$log(D_prob + 1e-8))
  
  MSE_loss <- tf$reduce_mean((M * X - M * G_sample)**2) / tf$reduce_mean(M)
  
  D_loss <- D_loss_temp
  G_loss <- G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver <- tf$compat$v1$train$AdamOptimizer()$minimize(D_loss, 
                                                          var_list = theta_D)
  G_solver <- tf$compat$v1$train$AdamOptimizer()$minimize(G_loss, 
                                                          var_list = theta_G)
  
  ## Iterations
  sess <- tf$compat$v1$Session()
  sess$run(tf$compat$v1$global_variables_initializer())

  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  
  for (it in 1:n){ 
    # Sample batch
    batch_idx <- sample_batch_index(nRow, batch_size)
    X_mb <- norm_data[batch_idx, ]  
    R_mb <- as.matrix(data_r[batch_idx])  
    M_mb <- data_m[batch_idx, ]
    # Sample random vectors  
    Z_mb <- uniform_sampler(0, 0.01, batch_size, nCol) 
    # Sample hint vectors
    H_mb_temp <- binary_sampler(1 - hint_rate, batch_size, 1)
    H_mb <- R_mb * H_mb_temp
  
    # Combine random vectors with observed vectors
    X_mb <- M_mb * X_mb + (1-M_mb) * Z_mb 
  
    D_loss_curr <- sess$run(c(D_solver, D_loss_temp), 
                            feed_dict = dict(R = R_mb, X = X_mb, 
                                             H = H_mb, M = M_mb))[[2]]
    run_result <- sess$run(c(G_solver, G_loss_temp, MSE_loss), 
                           feed_dict = dict(X = X_mb, R = R_mb, 
                                            H = H_mb, M = M_mb))
    G_loss_curr <- run_result[[2]]
    
    MSE_loss_curr <- run_result[[3]]
    
    pb$tick(tokens = list(what = "GAIN   "))
    Sys.sleep(2 / 100)
  }
    ## Return imputed data      
  Z_mb <- uniform_sampler(0, 0.01, nRow, nCol) 
  M_mb <- data_m
  X_mb <- norm_data       
  X_mb <- M_mb * X_mb + (1-M_mb) * Z_mb 
  
  imputed_data <- sess$run(c(G_sample), feed_dict = dict(X = X_mb, R = as.matrix(data_r)))[[1]]
  
  imputed_data <- data_m * norm_data + (1 - data_m) * imputed_data
  # Renormalization
  imputed_data <- renormalize(imputed_data, norm_parameters)  
  # Rounding
  #imputed_data <- rounding(imputed_data, data)  
  
  return (imputed_data)
}


gain <- function(data, m = 5, batch_size = 128, 
                 hint_rate = 0.9, alpha = 10, n = 5000){
  imputed_data <- NULL
  for (i in 1:m){
    imputed_data[[i]] <- gain_imp(data, batch_size, hint_rate, alpha, n)
  }
  
  return (imputed_data)
}

