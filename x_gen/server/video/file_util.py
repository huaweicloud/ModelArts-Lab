import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUtil:
    """文件操作工具类"""

    @staticmethod
    def copy_file(src_path: str, dst_path: str) -> bool:
        """
        复制文件从源路径到目标路径

        Args:
            src_path: 源文件路径
            dst_path: 目标文件路径

        Returns:
            bool: 是否复制成功
        """
        try:
            src = Path(src_path)
            dst = Path(dst_path)

            if not src.exists():
                return False

            # 确保目标目录存在
            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.info(f"文件复制失败: {e}")  # noqa: G004
            return False


file_util = FileUtil()
