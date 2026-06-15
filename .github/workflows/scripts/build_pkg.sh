#!/bin/bash
set -e

echo "build package..."
build_time="$(date +%Y%m%d%H%M%S)"

mkdir -p llm_infer
cp -r ./* llm_infer/
rm -rf llm_infer/tests
{
    echo "Build Branch: branch=$(git branch --show-current)"
    echo "Build Commit: $(git rev-parse HEAD)"
    echo "Build Time: ${build_time}"
    echo "vLLM Version: ${VLLM_VERSION}"
    echo "vLLM Ascend Version: ${VLLM_ASCEND_VERSION}"
} >> llm_infer/version.info

version=$(grep -m1 'version=' setup.py | sed "s/.*version=\"\([^\"]*\)\".*/\1/" || echo "0.1.0")

zip -q -r AscendCloud-LLM-"${version}"-"${build_time}".zip llm_infer/
zip -q -r AscendCloud-"${version}"-"${build_time}".zip AscendCloud-LLM-*.zip
echo "build package success."