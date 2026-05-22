"""纯文本转录 Loader：将转录字符串或 Whisper segments 转换为标准 Document 对象。"""
from __future__ import annotations

from modular_rag.core.types import Document
from modular_rag.libs.loader.base_loader import BaseLoader


class TranscriptTextLoader(BaseLoader):
    """
    接受原始转录文本字符串或 Whisper segments 列表，输出 Document 对象。
    metadata 必须由调用方提供，至少包含 source_path。
    不依赖 FileIntegrityChecker（由上层 task 负责幂等控制）。
    """

    def load(self, path: str) -> Document:  # type: ignore[override]
        """
        BaseLoader 兼容接口：path 参数此处传入转录文本字符串（非文件路径）。
        推荐使用 load_text() 以传递完整 metadata。
        """
        return Document(
            id="transcript",
            text=str(path),
            metadata={"source_path": "transcript://inline", "images": []},
        )

    def load_text(self, text: str, metadata: dict) -> list[Document]:
        """
        带完整 metadata 的文本摄取入口（推荐使用）。

        Args:
            text: 转录文本字符串。
            metadata: 至少含 source_path；建议携带 video_id、owner_id 等业务字段。
        """
        if not str(text).strip():
            return []

        meta = dict(metadata)
        if "source_path" not in meta:
            raise ValueError("TranscriptTextLoader.load_text(): metadata must contain 'source_path'")
        meta.setdefault("images", [])

        doc_id = meta.get("video_id", "transcript")
        return [Document(id=str(doc_id), text=str(text), metadata=meta)]

    def load_segments(
        self,
        segments: list[dict],
        base_metadata: dict,
    ) -> list[Document]:
        """
        将 Whisper verbose_json["segments"] 列表转换为多个 Document。
        每个 Document 对应一个 segment，metadata 携带：
          - start_s / end_s: 秒数（float），用于帧匹配
          - time_range: "MM:SS-MM:SS" 字符串，用于 cited_sources 展示

        Args:
            segments: Whisper verbose_json["segments"] 列表，
                      每项格式：{"id": 0, "start": 12.5, "end": 14.2, "text": "..."}
            base_metadata: 公共 metadata（source_path、video_id、owner_id 等）。
        """
        if not base_metadata.get("source_path"):
            raise ValueError("load_segments(): base_metadata must contain 'source_path'")

        docs: list[Document] = []
        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            meta = {
                **base_metadata,
                "start_s": start,
                "end_s": end,
                "time_range": f"{self._fmt_time(start)}-{self._fmt_time(end)}",
                "images": [],
            }
            video_id = base_metadata.get("video_id", "video")
            docs.append(Document(
                id=f"{video_id}_{int(start * 1000)}",
                text=text,
                metadata=meta,
            ))
        return docs

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """把秒数转为 MM:SS 或 HH:MM:SS 字符串。"""
        s = int(seconds)
        h, m, s_rem = s // 3600, (s % 3600) // 60, s % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s_rem:02d}"
        return f"{m:02d}:{s_rem:02d}"
