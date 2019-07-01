ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}"  )/.."  && pwd )"

pwd
echo "$ROOT"

python3 ${ROOT}/python/cli.py \
    --prog "deploy.Program" 0
