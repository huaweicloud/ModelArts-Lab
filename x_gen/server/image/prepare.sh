set -e
cd ..
pip install -r requirements.txt
cd ../x_base
pip install -e .
cd ../x_diffusers
pip install -e .
cd /home/ma-user
git clone https://gitcode.com/Ascend/MindIE-SD
cd MindIE-SD
git checkout d7eb2b550e4c217abb5d00726b5bb01e5e432059
python setup.py bdist_wheel
cd dist
pip install mindiesd-*.whl
