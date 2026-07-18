#!/usr/bin/env bash
set -x -euo pipefail

# refer to: https://github.com/numpy/numpy/pull/27472/files
pip install --force-reinstall numpy==1.26.4

# Find the installed numpy package path.
numpy_path="$(python -c 'import numpy, os; print(os.path.dirname(numpy.__path__[0]))')"
target_np_path="${numpy_path}/numpy/testing/_private/utils.py"

echo "target numpy path is: ${target_np_path}, fix bug of lscpu running start."

# Replace _SUPPORTS_SVE with check_support_sve in the exported list.
sed -i '41s/_SUPPORTS_SVE/check_support_sve/' "${target_np_path}"

# Patch check_support_sve to cache the lscpu result.
sed -i '
1239s/def check_support_sve():/def check_support_sve(__cache=[]):/
1244c\    if __cache:\n    \    return __cache[0]\n
1248s/return \x27sve\x27 in output.stdout/result = \x27sve\x27 in output.stdout/
1250s/return False/result = False/
1250a\    __cache.append(result)\n    return __cache[0]
' "${target_np_path}"

# Remove the eager check_support_sve call.
sed -i '/^_SUPPORTS_SVE = check_support_sve()$/d' "${target_np_path}"

echo "target numpy path is: ${target_np_path}, fix bug of lscpu running end."
