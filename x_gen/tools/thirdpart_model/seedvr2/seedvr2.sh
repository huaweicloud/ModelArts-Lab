set -e
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr
cd seedvr
git checkout 4df5cef5937001bcef056d5508c6a884849ffe13
git apply ../new_seedvr2.patch
pip install -e .
cd ..
