#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(Seurat); library(Matrix)})
a <- commandArgs(trailingOnly=TRUE)
if(length(a)<3) stop('Usage: Rscript R/export_pairedtag.R input.rds species output_dir')
obj <- readRDS(a[1]); species <- a[2]; out <- a[3]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(all(c('RNA') %in% Assays(obj)))
meta <- obj@meta.data; stopifnot(all(c('nCount_RNA','nCount_peaks','sample') %in% names(meta)))
keep <- !is.na(meta$nCount_RNA) & !is.na(meta$nCount_peaks) & meta$nCount_RNA>=750 & meta$nCount_peaks>=750 & meta$sample!='multiplet'
obj <- subset(obj,cells=rownames(meta)[keep]); meta <- obj@meta.data
if(species=='Pero') {male <- c('PL_M_1','PL_F_2'); female <- c('PL_F_1','PL_M_2'); xist <- 'LOC114708611'} else if(species=='Mus') {male <- c('6_1_M_R1','7_1_M_R2'); female <- c('8_1_F_R1','9_1_F_R2'); xist <- 'ENSMUSG00000086503'} else stop('species must be Pero or Mus')
meta$sex <- ifelse(meta$sample %in% male,'Male',ifelse(meta$sample %in% female,'Female',NA_character_))
if(anyNA(meta$sex)) stop('Unrecognized sample IDs; verify sample metadata')
rna <- GetAssayData(obj,assay='RNA',layer='counts')
if(!xist %in% rownames(rna)) stop(paste('Xist gene not found:',xist,'; inspect gene IDs'))
meta$xist_positive <- as.numeric(rna[xist,rownames(meta)])>0
meta$cell <- rownames(meta); meta$species <- species
write.csv(meta,file.path(out,paste0(species,'_metadata.csv')),row.names=FALSE)
Matrix::writeMM(rna,file.path(out,paste0(species,'_rna.mtx')))
writeLines(rownames(rna),file.path(out,paste0(species,'_genes.txt')))
writeLines(colnames(rna),file.path(out,paste0(species,'_cells.txt')))
peak_assays <- grep('peak|atac|h3k',Assays(obj),ignore.case=TRUE,value=TRUE)
if(length(peak_assays)==0) stop('No H3K27me3 peak assay found')
peak <- GetAssayData(obj,assay=peak_assays[1],layer='counts')
Matrix::writeMM(peak,file.path(out,paste0(species,'_h3k.mtx')))
writeLines(rownames(peak),file.path(out,paste0(species,'_peaks.txt')))
writeLines(colnames(peak),file.path(out,paste0(species,'_peak_cells.txt')))
message('Exported ',species,': ',ncol(rna),' cells, ',nrow(rna),' RNA genes, ',nrow(peak),' peaks')
