# Excuse the bad naming conventions 

# discrete uniform
n <- round(runif(1000)*10)
t <- table(n)
barplot(t)

# poisson 
n <- rpois(1000,3)
t <- table(n)
barplot(t)

# binomial 
n <- rbinom(1000,10, 0.2)
t <- table(n)
barplot(t)

# geometric
n <- rgeom(1000, 0.4)
n <- table(n)
barplot(n)

# exponential 
n <- round(rexp(1000, 0.1),1)
n <- table(n)
barplot(n)

# normal 
n <- round(rnorm(1000, 0, 1),1)
n <- table(n)
barplot(n)