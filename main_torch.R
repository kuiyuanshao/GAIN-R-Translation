library(torch)
library(progress)
source("utils_torch.R")
source("gain_torch.R")
sim <- function(mr){
  n  <- 500
  n2 <- 500 - 500*mr
  
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

output <- sim(mr = 0.9)
data <- output[[1]]
data_full <- output[[2]]


coef <- sd <- NULL
for (i in 1:5){
  imp <- gain(data)
  mod <- lm(Y ~ X + Z, data = imp)
  coef[i] <- coef(mod)[2]
  sd[i] <- sqrt(vcov(mod)[2, 2])
}
coef
mean(coef)
sd
mean(sd) + (5 + 1) * var(coef) / 5

modfull <- lm(Y ~ X + Z, data = data_full)
summary(modfull)

modmis <- lm(Y ~ X + Z, data = data)
summary(modmis)


load("SampleData_0001.RData")
library(survey)
library(data.table)
data <- samp <- as.data.table(samp) 
samp.solnas <- samp[(solnas==T),]
lm.lsodi <- glm(c_ln_na_bio1 ~ c_age + c_bmi + c_ln_na_avg + high_chol + usborn + female + bkg_pr + bkg_o,
                data=samp.solnas)
samp$c_ln_na_calib <- as.matrix(predict(lm.lsodi, newdata=samp, type = 'response'))
samp <- samp %>% dplyr::select(-matches("true")) %>% dplyr::select(-matches("bio"))
samp$c_ln_na_bio1[samp$solnas == F] <- NA
samp$c_ln_na_bio1[samp$solnas == T] <- data$c_ln_na_bio1[samp$solnas == T]

modcoef_hyper <- modvar_hyper <- modcoef_sbp <- modvar_sbp <- matrix(rep(0, 9 * 3), ncol = 9)
for (i in 1:3){
  imp <- gain(samp)
  mi.design <- svydesign(id=~BGid, strata=~strat, weights=~bghhsub_s2, data=imp)
  
  mod_hyper <- svyglm(hypertension ~ c_age + c_bmi + c_ln_na_bio1 + 
                        high_chol + usborn + female + bkg_pr + bkg_o,design=mi.design,family=quasibinomial())
  mod_sbp <- svyglm(sbp ~ c_age + c_bmi + c_ln_na_bio1 + 
                      high_chol + usborn + female + bkg_pr + bkg_o,design=mi.design,family=gaussian())
  
  modcoef_hyper[i, ] <- mod_hyper$coeff
  modcoef_sbp[i, ] <- mod_sbp$coeff
  
  modvar_hyper[i, ] <- summary(mod_hyper)$coeff[, 2]^2
  modvar_sbp[i, ] <- summary(mod_sbp)$coeff[, 2]^2
}
