import logging
import os
import time
from pathlib import Path

from obs import ObsClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
scc_path = Path(BASE_DIR) / "scc.conf"
SC_PASSWORD_FILE = "/scc/"


def get_single_file(dir_path):
    files = os.listdir(dir_path)

    filename = files[0]
    filepath = os.path.join(dir_path, filename)
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    return filename, content


ak_name = None
sk_password = None
if os.environ.get("OBS_BUCKET_NAME"):
    ak_name, sk_password = get_single_file(SC_PASSWORD_FILE)


class ObsStorageClient:
    def __init__(
        self,
        ak: str = ak_name,
        sk: str = sk_password,
        endpoint: str = os.environ.get("OBS_URL_PREFIX"),
        bucket: str = os.environ.get("OBS_BUCKET_NAME"),
    ):
        """
        通用存储客户端
        :param ak: Access Key
        :param sk: Secret Key
        :param endpoint: 服务地址 (华为云: https://obs.cn-north-4.myhuaweicloud.com)
        :param bucket: 存储桶名称
        """
        self.bucket = bucket
        print(f"ObsStorageClient 【endpoint】：{endpoint}")
        print(f"ObsStorageClient 【bucket】：{bucket}")
        self.client = ObsClient(access_key_id=ak, secret_access_key=sk, server=endpoint)

    def generate_url(self, filename: str, expire_seconds: int = os.environ.get("OBS_URL_EXPIRE_SECONDS")) -> str:
        """
        生成临时访问链接
        :param filename: 文件名
        :param expire_seconds: 有效期（秒），默认为配置中的OBS_URL_EXPIRE_SECONDS
        :return: 临时访问 URL
        """
        logger.info("start generate_url")
        file_path = os.path.join(os.environ.get("OBS_STORAGE_PATH"), filename)
        signed_url = self.client.createSignedUrl(
            method="GET", bucketName=self.bucket, objectKey=file_path, expires=expire_seconds
        )
        logger.info("finish generate_url")
        return signed_url["signedUrl"]

    def wait_object_ready(self, file_path, timeout=10):
        """
        等待对象存储文件可访问(需保证容器内可访问外网)
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.client.getObjectMetadata(self.bucket, file_path)
                if resp.status < 300:
                    return True
            except Exception as e:
                logger.warning(f"获取对象元数据失败: {e}, 继续等待...")  # noqa: G004
                pass  # 捕获异常后继续循环
            time.sleep(0.3)
        logger.warning("等待对象存储文件可访问超时")
        return False

    def save_to_obs(self, image_name, image):
        """
        保存文件到OBS
        :param image_name: 要上传的文件名
        :param image: 要上传的图片 PIL Image
        """
        logger.info("start save_to_obs")
        file_path = os.path.join(os.environ.get("OBS_STORAGE_PATH"), image_name)
        self.client.putObject(
            bucketName=self.bucket,
            objectKey=file_path,
            content=image,
        )

    def close(self):
        """
        关闭OBS客户端连接
        """
        if self.client:
            self.client.close()

    def __enter__(self):
        """
        上下文管理器入口
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出时自动关闭连接
        """
        self.close()
