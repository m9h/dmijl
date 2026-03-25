#!/bin/bash
# WAND CHARMED preprocessing with FSL (CUDA eddy on DGX Spark)
#
# Pipeline: merge b0s → topup → eddy (CUDA) → brain mask
# Uses fsl_sub for Slurm batching where applicable

set -e

SUB=sub-00395
SES=ses-02
DATADIR=/home/mhough/dev/dmipy/data/wand/${SUB}/${SES}/dwi
OUTDIR=/data/datasets/wand/${SUB}/${SES}/dwi
mkdir -p ${OUTDIR}

# Copy bval/bvec
cp ${DATADIR}/${SUB}_${SES}_acq-CHARMED_dir-AP_part-mag_dwi.bval ${OUTDIR}/
cp ${DATADIR}/${SUB}_${SES}_acq-CHARMED_dir-AP_part-mag_dwi.bvec ${OUTDIR}/
cp ${DATADIR}/${SUB}_${SES}_acq-CHARMED_dir-PA_part-mag_dwi.bval ${OUTDIR}/
cp ${DATADIR}/${SUB}_${SES}_acq-CHARMED_dir-PA_part-mag_dwi.bvec ${OUTDIR}/

DWI_AP=${DATADIR}/${SUB}_${SES}_acq-CHARMED_dir-AP_part-mag_dwi.nii.gz
DWI_PA=${DATADIR}/${SUB}_${SES}_acq-CHARMED_dir-PA_part-mag_dwi.nii.gz

echo "=== WAND CHARMED Preprocessing ==="
echo "Subject: ${SUB}"
echo "Input: ${DWI_AP} ($(du -h ${DWI_AP} | cut -f1))"
echo "Output: ${OUTDIR}"
echo ""

# Step 1: Extract b0 volumes for topup
echo "[1/5] Extracting b0 volumes..."
BVALS=$(cat ${OUTDIR}/${SUB}_${SES}_acq-CHARMED_dir-AP_part-mag_dwi.bval)
# Find first b0 index
B0_IDX=$(echo $BVALS | tr ' ' '\n' | grep -n "^0$" | head -1 | cut -d: -f1)
B0_IDX=$((B0_IDX - 1))  # fslroi is 0-indexed

fslroi ${DWI_AP} ${OUTDIR}/b0_AP ${B0_IDX} 1
fslroi ${DWI_PA} ${OUTDIR}/b0_PA 0 1
fslmerge -t ${OUTDIR}/b0_AP_PA ${OUTDIR}/b0_AP ${OUTDIR}/b0_PA
echo "  Done: b0_AP_PA.nii.gz"

# Step 2: Create acquisition parameters file
echo "[2/5] Creating acqparams.txt..."
# AP = anterior-posterior (phase encode y+), PA = posterior-anterior (y-)
# Readout time needs to be calculated from JSON or estimated
# Typical for Connectom: ~0.05s total readout
cat > ${OUTDIR}/acqparams.txt << 'ACQP'
0 1 0 0.05
0 -1 0 0.05
ACQP
echo "  Done"

# Step 3: Topup (distortion correction field estimation)
echo "[3/5] Running topup..."
topup --imain=${OUTDIR}/b0_AP_PA \
      --datain=${OUTDIR}/acqparams.txt \
      --config=b02b0.cnf \
      --out=${OUTDIR}/topup_results \
      --iout=${OUTDIR}/b0_topup \
      --fout=${OUTDIR}/topup_field \
      -v
echo "  Done"

# Step 4: Brain mask from topup-corrected b0
echo "[4/5] Brain extraction..."
fslmaths ${OUTDIR}/b0_topup -Tmean ${OUTDIR}/b0_topup_mean
bet ${OUTDIR}/b0_topup_mean ${OUTDIR}/b0_brain -m -f 0.3
echo "  Done: b0_brain_mask.nii.gz"

# Step 5: Eddy (CUDA) — motion + eddy current correction
echo "[5/5] Running eddy (CUDA)..."
# Create index file (all volumes use AP encoding = line 1 of acqparams)
NVOLS=$(fslnvols ${DWI_AP})
printf '1 %.0s' $(seq 1 ${NVOLS}) > ${OUTDIR}/index.txt

eddy_cuda11.0 \
    --imain=${DWI_AP} \
    --mask=${OUTDIR}/b0_brain_mask \
    --acqp=${OUTDIR}/acqparams.txt \
    --index=${OUTDIR}/index.txt \
    --bvecs=${OUTDIR}/${SUB}_${SES}_acq-CHARMED_dir-AP_part-mag_dwi.bvec \
    --bvals=${OUTDIR}/${SUB}_${SES}_acq-CHARMED_dir-AP_part-mag_dwi.bval \
    --topup=${OUTDIR}/topup_results \
    --out=${OUTDIR}/eddy_corrected \
    --repol \
    --data_is_shelled \
    -v

echo ""
echo "=== Preprocessing complete ==="
echo "Output: ${OUTDIR}/eddy_corrected.nii.gz"
ls -lh ${OUTDIR}/eddy_corrected.nii.gz
