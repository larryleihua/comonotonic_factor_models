load("merged2.RData")
dat = cbind(boco2040f2$BMXWT, boco2040f2$BMXWAIST, boco2040f2$DXXLAFAT, boco2040f2$DXXLLFAT, boco2040f2$DXXTRFAT,
            boco2040f2$DXDLALE, boco2040f2$DXDLLLE, boco2040f2$DXDTRLE,
            boco2040f2$DXXLABMC, boco2040f2$DXXLLBMC, boco2040f2$DXXTSBMC, boco2040f2$DXXPEBMC)
UU = data.frame(uscore(dat))
names(UU) = c("WT","WAIST","LAFAT","LLFAT","TRFAT","LALE","LLLE","TRLE","LABMC","LLBMC","TSBMC","PEBMC")
