library("mgcv")
#library("mgcViz")
#library("gratia")
library("ggsci")
library("gamair")
library(doParallel)
library(openxlsx)
library(dplyr)
library(readxl)

rm(list=ls())
StColname="HarmZ_score3rd.Ventricle"
EndColname="HarmZ_scoreLeft.TTG.transverse.temporal.gyrus" 

#parallel
cores =2
cl <- makeCluster(cores)
registerDoParallel(cl)


df_pMCI=read.xlsx("./pMCI_Fix_HarmZ_JADNI.xlsx")
df_pMCI$PTGENDER=as.factor(df_pMCI$PTGENDER)
df_pMCI$ID=as.factor(df_pMCI$ID)

## Get ROInames
dataStCol=which(colnames(df_pMCI)==StColname)
dataEndCol=which(colnames(df_pMCI)==EndColname)
ROInames=colnames(df_pMCI)[dataStCol:dataEndCol]

##GAMM fitting
gamm.model=list()
#results<-foreach(roi=ROInames,.packages="mgcv")%dopar%{
roi="HarmZ_scoreLeft.Hippocampus"
for(roi in ROInames){
  print(roi)
  #gamm_eq=as.formula(paste0(roi,"~ s(YearsToOnset) + ti(YearsToOnset, PTGENDER, bs = 'fs') + PTGENDER + s(ID, bs = 're')"))
  # gam(eTIV ~ s(ageatMRI, k = 3) + ti(ageatMRI, by = sex, bs = 'fs', k = 3) + sex, data = fs2, method = "REML")
  
  gamm_eq=as.formula(paste0(roi,"~ s(YearsToOnset)  + s(ID, bs = 're')"))
  # k = 3 because GAM and mostly adults
  gamm_results=gam(gamm_eq,data=df_pMCI,method="REML")
  gamm.model[[roi]]=gamm_results  
}
objname="gamm_pMCI.obj"
saveRDS(gamm.model,objname)
  #"data = fs.c2, method = 'REML')" 
  
