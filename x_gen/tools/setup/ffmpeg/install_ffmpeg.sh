git clone https://github.com/mirror/x264.git
cd x264
./configure --prefix=/usr/local --enable-shared --enable-static --disable-asm
make -j64
sudo make install
sudo ldconfig

wget https://ffmpeg.org/releases/ffmpeg-4.2.11.tar.gz
tar -xvf ffmpeg-4.2.11.tar.gz
cd ffmpeg-4.2.11
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"
./configure --enable-shared --prefix=/usr/local/ffmpeg  --enable-libx264 --enable-gpl --enable-nonfree

make -j64
sudo make install
sudo sed -i "1i /usr/local/lib"  /etc/ld.so.conf
sudo ldconfig
