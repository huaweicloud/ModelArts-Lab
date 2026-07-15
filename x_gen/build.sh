cd x_base
pip wheel --no-deps -w dist . -v
mv dist/*.whl ../
rm -r build dist
cd ..
cd x_diffusers
pip wheel --no-deps -w dist . -v
mv dist/*.whl ../
rm -r build dist
cd ..
