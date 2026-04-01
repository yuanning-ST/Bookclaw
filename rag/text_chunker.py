"""
中文文本分块器
基于 HanLP 中文分词，支持句子边界感知的智能分块
"""
import hanlp
import re
from typing import List


class ChineseTextChunker:
    """中文文本分块器，将长文本分割成带有重叠的文本块"""

    def __init__(self, chunk_size: int = 500, overlap: int = 100, max_text_length: int = 500000):
        """
        初始化分块器

        Args:
            chunk_size: 每个文本块的目标大小（tokens 数量）
            overlap: 相邻文本块的重叠大小（tokens 数量）
            max_text_length: HanLP 处理的最大文本长度，超过此长度将进行预分割
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size 必须大于 overlap")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

    def _preprocess_large_text(self, text: str) -> List[str]:
        """预处理过大的文本，将其分割成较小的段落"""
        if len(text) <= self.max_text_length:
            return [text]

        target_segment_size = min(self.max_text_length, max(10000, self.max_text_length // 2))

        # 首先按段落分割
        paragraphs = text.split('\n\n')
        if len(paragraphs) < 5:
            paragraphs = text.split('\n')

        # 重新组合段落，确保每个段落不超过目标大小
        processed_segments = []
        current_segment = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > target_segment_size:
                if current_segment:
                    processed_segments.append(current_segment)
                    current_segment = ""
                split_paras = self._split_long_paragraph(para, target_segment_size)
                processed_segments.extend(split_paras)
            else:
                if len(current_segment) + len(para) + 2 > target_segment_size:
                    if current_segment:
                        processed_segments.append(current_segment)
                    current_segment = para
                else:
                    if current_segment:
                        current_segment += "\n\n" + para
                    else:
                        current_segment = para

        if current_segment:
            processed_segments.append(current_segment)

        return processed_segments

    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        """分割超长段落"""
        if len(text) <= max_size:
            return [text]

        # 按句子分割
        sentences = re.split(r'([.!?.!?])', text)

        # 重新组合句子和标点
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence.strip():
                combined_sentences.append(sentence + punctuation)

        # 如果没有找到句子边界，按固定长度分割
        if not combined_sentences:
            return [text[i:i + max_size] for i in range(0, len(text), max_size)]

        # 重新组合句子，确保不超过最大长度
        segments = []
        current_segment = ""

        for sentence in combined_sentences:
            if len(sentence) > max_size:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                for i in range(0, len(sentence), max_size):
                    segments.append(sentence[i:i + max_size])
            else:
                if len(current_segment) + len(sentence) > max_size:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += sentence

        if current_segment:
            segments.append(current_segment)

        return segments

    def _safe_tokenize(self, text: str) -> List[str]:
        """安全的分词方法，处理可能的异常"""
        try:
            if len(text) > self.max_text_length:
                return list(text)
            tokens = self.tokenizer(text)
            return tokens if tokens else []
        except Exception:
            return list(text)

    def chunk_text(self, text: str) -> List[List[str]]:
        """将文本分割成块，返回 token 列表的列表"""
        if not text or len(text) < self.chunk_size / 10:
            tokens = self._safe_tokenize(text)
            return [tokens] if tokens else []

        # 预处理过大文本
        text_segments = self._preprocess_large_text(text)

        # 处理每个文本段落
        all_chunks = []
        for segment in text_segments:
            segment_chunks = self._chunk_single_segment(segment)
            all_chunks.extend(segment_chunks)

        return all_chunks

    def _chunk_single_segment(self, text: str) -> List[List[str]]:
        """处理单个文本段落的分块"""
        if not text:
            return []

        all_tokens = self._safe_tokenize(text)
        if not all_tokens:
            return []

        chunks = []
        start_pos = 0

        while start_pos < len(all_tokens):
            end_pos = min(start_pos + self.chunk_size, len(all_tokens))

            # 如果不是最后一块，尝试在句子边界结束
            if end_pos < len(all_tokens):
                sentence_end = self._find_next_sentence_end(all_tokens, end_pos)
                if sentence_end <= start_pos + self.chunk_size + 100:
                    end_pos = sentence_end

            chunk = all_tokens[start_pos:end_pos]
            if chunk:
                chunks.append(chunk)

            if end_pos >= len(all_tokens):
                break

            overlap_start = max(start_pos, end_pos - self.overlap)
            next_sentence_start = self._find_previous_sentence_end(all_tokens, overlap_start)

            if next_sentence_start > start_pos and next_sentence_start < end_pos:
                start_pos = next_sentence_start
            else:
                start_pos = overlap_start

            if start_pos >= end_pos:
                start_pos = end_pos

        return chunks

    def _is_sentence_end(self, token: str) -> bool:
        """判断 token 是否为句子结束符"""
        return token in ['.', '!', '?']

    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向后查找句子结束位置"""
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)

    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向前查找句子结束位置"""
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0

    def tokens_to_text(self, tokens: List[str]) -> str:
        """将 token 列表转换回文本"""
        return ''.join(tokens)

    def chunk_text_to_strings(self, text: str) -> List[str]:
        """将文本分割成字符串块（方便使用）"""
        token_chunks = self.chunk_text(text)
        return [self.tokens_to_text(chunk) for chunk in token_chunks]
