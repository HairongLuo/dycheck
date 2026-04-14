#!/bin/bash
#SBATCH --output=/cluster/home/hailuo/project/shape-of-motion/logs/generate_covisibles/slurm-%j.out
#SBATCH --nodes=1             # Keep it to 1 node unless scaling out
#SBATCH --ntasks=1            # Single training process
#SBATCH --mem-per-cpu=12500M      # Request memory per CPU
#SBATCH --gpus=rtx_4090:1     # Use all 4 RTX 4090 GPUs
#SBATCH --cpus-per-task=4    # Allocate more CPUs for data loading
#SBATCH --time=4:00:00       # Set a reasonable training time limit

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_name>"
    exit 1
fi

DATA_NAME=$1

module load stack/2024-04 gcc/8.5.0
module load cuda/11.8.0
module load cudnn/8.2.0.53-11.3
module load eth_proxy

source /cluster/project/cvg/students/hailuo/miniconda3/etc/profile.d/conda.sh
conda activate dycheck

# copy data from project dir
data_dir=/cluster/project/cvg/students/hailuo/shape-of-motion/data_root/iphone/$DATA_NAME/
dycheck_dir=/cluster/home/hailuo/project/dycheck/
dycheck_data_dir=$dycheck_dir/datasets/mycustom/$DATA_NAME

# check if the source data exists
if [ ! -d "$data_dir" ]; then
    echo "Error: Data directory not found"
    exit 1
fi

# copy data to dycheck dataset dir
rgb_dir=$dycheck_data_dir/rgb
mkdir -p $rgb_dir
cp -r $data_dir/rgb/1x/ $rgb_dir
cp -r $data_dir/splits/ $dycheck_data_dir

# generate covisible masks
cd /cluster/home/hailuo/project/dycheck

# ! process_covisible.Config.chunk in process_covisible.gin cannot be set to arbitrary number, 
# ! as some choices may lead to cuDNN error.
python tools/process_covisible_mycustom.py \
    --gin_configs 'configs/mycustom/process_covisible.gin' \
    --gin_bindings "SEQUENCE=\"${DATA_NAME}\""

# copy result to data dir
mkdir -p $data_dir/flow3d_preprocessed/covisible/1x/
cp -r $dycheck_data_dir/covisible/1x/val/ $data_dir/flow3d_preprocessed/covisible/1x/

# clean up
rm -rf $dycheck_data_dir