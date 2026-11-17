args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2) stop("Supply output working directory and original SEM archive directory")
setwd(args[1])
suppressPackageStartupMessages(library(lavaan))
archive <- args[2]
input_name <- readLines(file.path(archive,"source_file.txt"),encoding="UTF-8",warn=FALSE)[1]
is_absolute <- startsWith(input_name, "/") || grepl("^[A-Za-z]:", input_name) || startsWith(input_name, "\\\\")
if (!is_absolute) input_name <- file.path(archive, input_name)
df <- read.csv(input_name)
vars <- c("BMSI","Wind","Shade","DSR","Received","DayLST","ST")
stopifnot(nrow(df)==622,!anyDuplicated(df$city_id),all(is.finite(as.matrix(df[vars]))))
z <- scale(df[,vars],scale=FALSE); z <- as.data.frame(sweep(z,2,sqrt(colMeans(z^2)),"/"))
base <- readLines(file.path(archive,"models/original_paths__original.txt"))
reduced <- sub(" + h*BMSI","",base,fixed=TRUE)
stopifnot(sum(base!=reduced)==1,"ST ~ t*DayLST + g*Received + k*Wind" %in% reduced)
mods <- list(full=base,drop_BMSI_ST=reduced)
paths <- list()
for(k in names(mods)) for(est in c("ML","MLR")){
 id <- paste(k,est,sep="_")
 f <- sem(paste(mods[[k]],collapse="\n"),data=z,estimator=est,ov.order="data",fixed.x=FALSE,
   meanstructure=FALSE,auto.cov.y=FALSE,auto.cov.lv.x=FALSE,control=list(iter.max=10000,eval.max=20000))
 stopifnot(lavInspect(f,"converged"),lavInspect(f,"post.check"),min(eigen(lavInspect(f,"information"),symmetric=TRUE)$values)>0)
 p <- standardizedSolution(f);raw <- parameterEstimates(f)
                                                                           
                                                                                      
 pairs <- paste(raw$lhs,raw$op,raw$rhs)
 p$p_raw <- raw$pvalue[match(paste(p$lhs,p$op,p$rhs),pairs)]
 paths[[id]] <- cbind(model=k,estimator=est,p)
}
p <- do.call(rbind,paths)
for(k in names(mods)){a<-subset(p,model==k & estimator=="ML");b<-subset(p,model==k & estimator=="MLR");stopifnot(max(abs(a$est.std-b$est.std))<1e-6)}
prior<-read.csv(file.path(archive,"standardized_parameters.csv"))
a<-subset(p,model=="full" & estimator=="MLR" & op=="~");b<-subset(prior,family=="original_paths" & variant=="original" & op=="~")
stopifnot(nrow(a)==11,max(abs(a$est.std-b$est.std))<1e-6)
write.csv(p,"paths.csv",row.names=FALSE);writeLines(reduced,"model.txt")
print(subset(p,lhs=="ST" & op=="~" & rhs=="DayLST",select=c(model,estimator,est.std,se,pvalue,p_raw,ci.lower,ci.upper)),digits=5)
