"""
文件读取器 - 支持多种文档格式
"""
import codecs
import os
from typing import List, Tuple, Dict, Optional
import PyPDF2
from docx import Document
import csv
import json
import yaml
from yaml import CLoader as Loader


class FileReader:
    """
    文件读取器，支持多种文件格式
    """

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def read_files(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Tuple[str, str]]:
        """
        读取指定扩展名的文件

        Args:
            file_extensions: 文件扩展名列表，如不指定则读取所有支持的格式
            recursive: 是否递归读取子目录

        Returns:
            List[Tuple[str, str]]: (文件路径，内容) 元组列表
        """
        supported_extensions = {
            '.txt': self._read_txt,
            '.pdf': self._read_pdf,
            '.md': self._read_markdown,
            '.docx': self._read_docx,
            '.csv': self._read_csv,
            '.json': self._read_json,
            '.yaml': self._read_yaml,
            '.yml': self._read_yaml,
        }

        if file_extensions is None:
            file_extensions = list(supported_extensions.keys())

        results = []

        try:
            if recursive:
                results = self._read_files_recursive(self.directory_path, file_extensions, supported_extensions)
            else:
                for item in os.listdir(self.directory_path):
                    item_path = os.path.join(self.directory_path, item)
                    if os.path.isfile(item_path):
                        file_ext = os.path.splitext(item)[1].lower()
                        if file_ext in file_extensions and file_ext in supported_extensions:
                            content = supported_extensions[file_ext](item_path)
                            results.append((item, content))
        except Exception as e:
            print(f"读取文件时出错：{str(e)}")

        return results

    def _read_files_recursive(self, root_dir: str, file_extensions: List[str], supported_extensions: Dict) -> List[Tuple[str, str]]:
        """递归读取目录及其子目录中的文件"""
        results = []

        try:
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)

                if os.path.isdir(item_path):
                    sub_results = self._read_files_recursive(item_path, file_extensions, supported_extensions)
                    results.extend(sub_results)
                elif os.path.isfile(item_path):
                    file_ext = os.path.splitext(item)[1].lower()
                    if file_ext in file_extensions:
                        rel_path = os.path.relpath(item_path, self.directory_path)
                        if file_ext in supported_extensions:
                            try:
                                content = supported_extensions[file_ext](item_path)
                                results.append((rel_path, content))
                            except Exception as e:
                                print(f"读取文件 {rel_path} 时出错：{str(e)}")
        except Exception as e:
            print(f"列出目录 {root_dir} 时出错：{str(e)}")

        return results

    def _read_txt(self, file_path: str) -> str:
        """读取 TXT 文件"""
        try:
            with codecs.open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
        except Exception as e:
            return f"[无法读取文件内容：{str(e)}]"

    def _read_pdf(self, file_path: str) -> str:
        """读取 PDF 文件"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
            return text
        except Exception as e:
            return f"[无法读取 PDF 文件内容：{str(e)}]"

    def _read_markdown(self, file_path: str) -> str:
        """读取 Markdown 文件"""
        try:
            with codecs.open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
        except Exception as e:
            return f"[无法读取 Markdown 文件内容：{str(e)}]"

    def _read_docx(self, file_path: str) -> str:
        """读取 Word 文档"""
        try:
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"[无法读取 Word 文档内容：{str(e)}]"

    def _read_csv(self, file_path: str) -> str:
        """读取 CSV 文件"""
        try:
            text = []
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    text.append(','.join(row))
            return '\n'.join(text)
        except Exception as e:
            return f"[无法读取 CSV 文件内容：{str(e)}]"

    def _read_json(self, file_path: str) -> str:
        """读取 JSON 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                data = json.load(file)
                return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"[无法读取 JSON 文件内容：{str(e)}]"

    def _read_yaml(self, file_path: str) -> str:
        """读取 YAML 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                data = yaml.load(file, Loader=Loader)
                return yaml.dump(data, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            return f"[无法读取 YAML 文件内容：{str(e)}]"

    def read_file_by_path(self, file_path: str) -> str:
        """
        根据文件路径读取单个文件

        Args:
            file_path: 文件完整路径

        Returns:
            str: 文件内容
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        supported_extensions = {
            '.txt': self._read_txt,
            '.pdf': self._read_pdf,
            '.md': self._read_markdown,
            '.docx': self._read_docx,
            '.csv': self._read_csv,
            '.json': self._read_json,
            '.yaml': self._read_yaml,
            '.yml': self._read_yaml,
        }

        if file_ext in supported_extensions:
            return supported_extensions[file_ext](file_path)
        else:
            return f"[不支持的文件格式：{file_ext}]"
