ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}"  )/.."  && pwd )"

pwd
echo "$ROOT"

python3 ${ROOT}/python/cli.py \
    --prog "dog_and_cat_train.Program" 0 \
    --max_epochs 30 \
    --batch_size 32 \
    --num_gpus 1 \
    --dataset_name "dog_and_cat_200"
