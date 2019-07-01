FROM dls.io/dls/custom-gpu-cuda9-base-inner:1.0-latest

ARG download

ENV PATH /root/miniconda2/bin:$PATH
ENV LD_LIBRARY_PATH /root/miniconda2/lib:$LD_LIBRARY_PATH

RUN mkdir ~/install && cd ~/install && \
  wget $download/dls-release/ubuntu-16.04/third_party/Miniconda/Miniconda2-4.5.1-Linux-x86_64.sh && \
  bash Miniconda2-4.5.1-Linux-x86_64.sh -b -p /root/miniconda2 && \
  mv ~/.pip/pip.conf ~/.pip/pip.conf~ && \
  wget $download/dls-release/ubuntu-16.04/ci-config/pip.conf && mv pip.conf ~/.pip/ && \
  wget -r -l1 -nd --reject "index.html*" $download/dls-release/ubuntu-16.04/moxing-tf-1.8/latest/ 2>/dev/null && \
  pip --no-cache-dir install moxing_framework-*-py2-*.whl && \
  pip --no-cache-dir install boto3==1.7.29 numpy==1.15.4 netifaces==0.10.7 pyzmq==17.0.0 && \
  rm -rf ~/.pip/pip.conf && \
  mv ~/.pip/pip.conf~ ~/.pip/pip.conf && \
  rm -rf ~/install

RUN rm -f ~/.pip/pip.conf && \
  echo "[global]" >> ~/.pip/pip.conf && \
  echo "trusted-host=10.93.238.51" >> ~/.pip/pip.conf && \
  echo "index-url=http://10.93.238.51/pypi/simple/" >> ~/.pip/pip.conf

