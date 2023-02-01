library(torch)

source("utils_torch.R")
source("gain_torch.R")
sim <- function(mr){
  n  <- 4000
  n2 <- 4000 - 4000*mr
  
  beta <- c(1, 1, 1)
  e_U <- c(sqrt(3),sqrt(3))
  mx <- 0; sx <- 1; zrange <- 1; zprob <- .5
  
  simZ   <- rbinom(n, zrange, zprob)
  simX   <- (1-simZ)*rnorm(n, 0, 1) + simZ*rnorm(n, 0.5, 1)
  #error
  epsilon <- rnorm(n, 0, 1)
  #Y is added with epsilon
  simY    <- beta[1] + beta[2]*simX + beta[3]*simZ + epsilon
  #make X_star depends on it
  simX_tilde <- simX + rnorm(n, 0, e_U[1]*(simZ==0) + e_U[2]*(simZ==1))
  data <- data.frame(Y_tilde=simY, X_tilde=simX_tilde, Y=simY, X=simX, Z=simZ)
  ##### Designs
  ## SRS
  id_phase2 <- c(sample(n, n2))
  data$X[-id_phase2] <- NA
  return (data)
}

data <- sim(mr = 0.4)

imp <- gain(data, 128, 0.9, 10, 5000)
mod <- lm(Y ~ X + Z, data = imp)

summary(mod)
