export INSTALL_DIR=$HOME/.local/

echo "load gcc and python"
module load gcc/8.2.0
module load python_gpu/3.8.5
module load cuda/11.7.0

if [ -f /cluster/scratch/horatan/CIL/bin/activate ];
then
    source /cluster/scratch/horatan/CIL/bin/activate;
fi

