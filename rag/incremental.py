"""
增量更新 - 检测新增/修改的文件并处理
"""
import os
import hashlib
import json
from typing import Dict, List, Set, Tuple


class FileHashManager:
    """文件哈希管理器，用于跟踪已处理文件的哈希值"""

    def __init__(self, hash_file: str = "./rag_data/file_hashes.json"):
        """
        初始化哈希管理器

        Args:
            hash_file: 哈希记录文件路径
        """
        self.hash_file = hash_file
        self._hashes: Dict[str, str] = {}
        self._load()

    def _load(self):
        """从文件加载哈希记录"""
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, "r", encoding="utf-8") as f:
                    self._hashes = json.load(f)
            except Exception as e:
                print(f"加载哈希文件失败：{e}")
                self._hashes = {}

    def _save(self):
        """保存哈希记录到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.hash_file) or ".", exist_ok=True)
        with open(self.hash_file, "w", encoding="utf-8") as f:
            json.dump(self._hashes, f, indent=2, ensure_ascii=False)

    def compute_file_hash(self, file_path: str) -> str:
        """
        计算文件的哈希值（MD5）

        Args:
            file_path: 文件路径

        Returns:
            str: 文件的MD5哈希值
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # 分块读取，避免大文件内存问题
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"计算文件哈希失败 {file_path}: {e}")
            return ""

    def get_stored_hash(self, file_path: str) -> str:
        """获取已存储的文件哈希值"""
        return self._hashes.get(file_path, "")

    def update_hash(self, file_path: str, hash_value: str):
        """更新文件的哈希记录"""
        self._hashes[file_path] = hash_value
        self._save()

    def remove_hash(self, file_path: str):
        """删除文件的哈希记录"""
        if file_path in self._hashes:
            del self._hashes[file_path]
            self._save()

    def get_all_files(self) -> Set[str]:
        """获取所有已记录的文件路径"""
        return set(self._hashes.keys())

    def clear(self):
        """清空所有哈希记录"""
        self._hashes = {}
        self._save()


class IncrementalUpdater:
    """增量更新器，检测并处理新增/修改的文件"""

    def __init__(
        self,
        files_dir: str = "./books",
        hash_file: str = "./rag_data/file_hashes.json",
    ):
        """
        初始化增量更新器

        Args:
            files_dir: 文档目录
            hash_file: 哈希记录文件路径
        """
        self.files_dir = files_dir
        self.hash_manager = FileHashManager(hash_file)

    def detect_new_files(self, file_extensions: List[str] = None) -> Tuple[List[str], List[str]]:
        """
        检测新增和修改的文件

        Args:
            file_extensions: 要检测的文件扩展名列表

        Returns:
            Tuple[List[str], List[str]]: (新增文件列表, 修改文件列表)
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.pdf', '.md', '.docx', '.csv', '.json', '.yaml', '.yml']

        new_files = []
        modified_files = []

        # 遍历目录中的所有文件
        for root, _, files in os.walk(self.files_dir):
            for filename in files:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in file_extensions:
                    continue

                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.files_dir)

                # 计算当前文件哈希
                current_hash = self.hash_manager.compute_file_hash(full_path)
                if not current_hash:
                    continue

                # 获取已存储的哈希
                stored_hash = self.hash_manager.get_stored_hash(rel_path)

                if not stored_hash:
                    # 新文件
                    new_files.append(rel_path)
                elif stored_hash != current_hash:
                    # 文件已修改
                    modified_files.append(rel_path)

        return new_files, modified_files

    def detect_all_changes(self, file_extensions: List[str] = None) -> Dict[str, List[str]]:
        """
        检测所有文件变化

        Args:
            file_extensions: 要检测的文件扩展名列表

        Returns:
            Dict: {"new": 新增文件, "modified": 修改文件, "unchanged": 未变化文件}
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.pdf', '.md', '.docx', '.csv', '.json', '.yaml', '.yml']

        new_files = []
        modified_files = []
        unchanged_files = []

        current_files = set()

        # 遍历目录中的所有文件
        for root, _, files in os.walk(self.files_dir):
            for filename in files:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in file_extensions:
                    continue

                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.files_dir)
                current_files.add(rel_path)

                # 计算当前文件哈希
                current_hash = self.hash_manager.compute_file_hash(full_path)
                if not current_hash:
                    continue

                # 获取已存储的哈希
                stored_hash = self.hash_manager.get_stored_hash(rel_path)

                if not stored_hash:
                    new_files.append(rel_path)
                elif stored_hash != current_hash:
                    modified_files.append(rel_path)
                else:
                    unchanged_files.append(rel_path)

        # 检测已删除的文件（在记录中但不在目录中）
        deleted_files = list(self.hash_manager.get_all_files() - current_files)

        return {
            "new": new_files,
            "modified": modified_files,
            "unchanged": unchanged_files,
            "deleted": deleted_files,
        }

    def mark_file_processed(self, file_path: str, full_path: str = None):
        """
        标记文件已处理（更新哈希记录）

        Args:
            file_path: 相对文件路径
            full_path: 完整文件路径（如不提供则自动构建）
        """
        if full_path is None:
            full_path = os.path.join(self.files_dir, file_path)

        hash_value = self.hash_manager.compute_file_hash(full_path)
        if hash_value:
            self.hash_manager.update_hash(file_path, hash_value)

    def remove_deleted_records(self, deleted_files: List[str]):
        """
        删除已不存在文件的哈希记录

        Args:
            deleted_files: 已删除的文件列表
        """
        for file_path in deleted_files:
            self.hash_manager.remove_hash(file_path)

    def get_stats(self) -> Dict:
        """获取哈希记录统计信息"""
        return {
            "total_recorded": len(self.hash_manager._hashes),
            "recorded_files": list(self.hash_manager._hashes.keys()),
        }