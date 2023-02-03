library(tensorflow)
library(progress)
tf$constant("Hello Tensorflow!")
x <- import("tensorflow.compat.v1") 
x$disable_v2_behavior()

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
  data_full <- data.frame(Y_tilde=simY, X_tilde=simX_tilde, Y=simY, X=simX, Z=simZ)
  ##### Designs
  ## SRS
  data <- data_full
  id_phase2 <- c(sample(n, n2))
  data$X[-id_phase2] <- NA
  data$R <- 0
  data$R[id_phase2] <- 1
  data$pi <- n2 / n
  return (list(data, data_full))
}

output <- sim(mr = 0.8)
data <- output[[1]]
data_full <- output[[2]]
imputed_data <- gain_imp(data, 128, 0.9, 10, 5000)
mod1 <- lm(Y ~ X + Z, data = imputed_data)
mod_full <- lm(Y ~ X + Z, data = data_full)


library(mixgb)

mixgb_d <- mixgb(data, m = 5)
coef_i <- se_i <- NULL
for (i in 1:5){
  mod <- lm(Y ~ X + Z, data = mixgb_d[[i]])
  coef_i[i] <- coef(mod)[2]
  se_i[i] <- sqrt(vcov(mod)[2, 2])
}
coef_i
se_i
summary(mod1)
summary(mod_full)

sim1 <- function(mr){
  n  <- 4000
  n2 <- 4000 - 4000*mr
  
  beta <- c(1, 1, 1)
  e_U <- c(sqrt(3),sqrt(3))
  mx <- 0; sx <- 1; zrange <- 1; zprob <- .5
  
  simZ   <- rbinom(n, zrange, zprob)
  simB   <- rbinom(n, 1, 0.3)
  simK   <- (1-simB)*rnorm(n, 0.5, 1) + simB*rnorm(n, 2, 1)
  simX   <- (1-simZ)*rnorm(n, 0, 1) + simZ*rnorm(n, 0.5, 1) + 0.2*simK
  epsilon <- rnorm(n, 0, 1)
  #Y is added with epsilon
  simY    <- beta[1] + beta[2]*simX + beta[3]*simZ + simK + epsilon
  #make X_star depends on it
  simX_tilde <- simX + rnorm(n, 0, e_U[1]*(simZ==0) + e_U[2]*(simZ==1))
  simK_tilde <- simK + rnorm(n, 0, e_U[1]*(simB==0) + e_U[2]*(simB==1))
  data_full <- data.frame(Y_tilde=simY, X_tilde=simX_tilde, 
                          K_tilde = simK_tilde,
                          Y=simY, X=simX, Z=simZ, B = simB, K = simK)
  ##### Designs
  ## SRS
  data <- data_full
  id_phase2 <- c(sample(n, n2))
  data$X[-id_phase2] <- NA
  data$K[-id_phase2] <- NA
  data$R <- 0
  data$R[id_phase2] <- 1
  data$pi <- n2 / n
  return (list(data, data_full))
}

output <- sim1(mr = 0.8)
data <- output[[1]]
data_full <- output[[2]]
imputed_data <- gain_imp(data, 128, 0.9, 10, 5000)
mod1 <- lm(Y ~ X + Z + K + B, data = imputed_data)
mod_full <- lm(Y ~ X + Z + K + B, data = data_full)


library(mixgb)

mixgb_d <- mixgb(data, m = 5)
coef_i <- se_i <- NULL
for (i in 1:5){
  mod <- lm(Y ~ X + Z + K + B, data = mixgb_d[[i]])
  coef_i[[i]] <- coef(mod)[c(2, 4)]
  se_i[[i]] <- sqrt(vcov(mod)[c(2, 4), c(2, 4)])
}

mod_mis <- lm(Y ~ X + Z + K + B, data = data)
summary(mod1)
summary(mod)
summary(mod_mis)
summary(mod_full)