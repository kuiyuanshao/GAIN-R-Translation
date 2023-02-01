



discriminator <- function(X, H, nCol){
  inputs <- torch_cat(list(X, H), dim = 1)
  model <- nn_sequential(
    nn_Linear(nCol * 2, nCol),
    nn_ReLu(),
    nn_Linear(nCol, nCol),
    nn.Sigmoid(),
    to(device = device)
  )
  return (model(inputs))
}

generator <- function(X, M, nCol){
  inputs <- torch_cat(list(X, M), dim = 1)
  model <- nn_sequential(
    nn_Linear(nCol * 2, nCol),
    nn_ReLu(),
    nn_Linear(nCol, nCol),
    nn.Sigmoid(),
    to(device = device)
  )
  return (model(inputs))
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