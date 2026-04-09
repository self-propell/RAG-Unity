"""
Retrieve relevant chunks from domain-specific Chroma collections.

Each project now uses multiple collections so code, scenes, prefabs, audio,
images, and generic assets can use different chunking and embedding profiles.
"""
import re
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from core import config


ALL_COLLECTION_DOMAINS = ("code", "scene", "prefab", "asset", "audio", "image")
EXTENSION_TO_DOMAIN = {
    ".cs": "code",
    ".asmdef": "code",
    ".unity": "scene",
    ".prefab": "prefab",
    ".asset": "asset",
    ".spriteatlas": "image",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tga": "image",
    ".psd": "image",
    ".exr": "image",
    ".wwu": "audio",
    ".bnk": "audio",
    ".wem": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".mp3": "audio",
}
CHUNK_TYPE_TO_DOMAINS = {
    "class": ("code",),
    "method": ("code",),
    "file": ("code",),
    "assembly": ("code",),
    "scene": ("scene",),
    "prefab": ("prefab",),
    "asset": ("asset",),
    "audio": ("audio",),
    "image": ("image",),
}
_EMBEDDING_CACHE: dict[str, object] = {}


@dataclass
class SearchResult:
    text: str
    relative_path: str
    file_path: str
    chunk_type: str
    class_name: str
    method_name: str
    namespace: str
    start_line: int
    score: float
    collection_domain: str = ""
    file_ext: str = ""
    relative_dir: str = "."
    top_level_dir: str = "."
    assembly_name: str = ""
    asset_kind: str = ""
    raw_score: float = 0.0
    rerank_score: float = 0.0
    parent_context: str = ""

    def format_for_prompt(self) -> str:
        location = self.relative_path
        if self.class_name:
            location += f" > {self.class_name}"
        if self.method_name:
            location += f".{self.method_name}"

        lines = [f"// [{self.chunk_type}] {location} (line {self.start_line})"]
        if self.collection_domain:
            lines.append(f"// domain: {self.collection_domain}")
        if self.assembly_name:
            lines.append(f"// assembly: {self.assembly_name}")
        if self.parent_context:
            lines.append(f"// parent context\n{self.parent_context}")
        lines.append(self.text)
        return "\n".join(lines)


@dataclass(frozen=True)
class QueryPlan:
    preferred_domains: tuple[str, ...]
    secondary_domains: tuple[str, ...]
    preferred_chunk_types: tuple[str, ...]
    deprioritized_chunk_types: tuple[str, ...]
    preferred_extensions: tuple[str, ...]
    path_terms: tuple[str, ...]
    symbol_terms: tuple[str, ...]
    assembly_terms: tuple[str, ...]
    candidate_k: int


def domain_for_extension(ext: str) -> str:
    return EXTENSION_TO_DOMAIN.get(ext.lower(), "asset")


def domains_for_chunk_type(chunk_type: str | None) -> tuple[str, ...]:
    if not chunk_type:
        return ALL_COLLECTION_DOMAINS
    return CHUNK_TYPE_TO_DOMAINS.get(chunk_type, ALL_COLLECTION_DOMAINS)


def collection_name_for_domain(base_collection_name: str, domain: str) -> str:
    return f"{base_collection_name}_{domain}"


def get_embedding_function(domain: str = "code"):
    profile = "code" if domain == "code" else "unity_asset"
    cached = _EMBEDDING_CACHE.get(profile)
    if cached is not None:
        return cached

    import os
    from pathlib import Path

    local_dir = Path(__file__).resolve().parent.parent / "models"
    local_candidates = []
    remote_candidates = []

    if profile == "code":
        local_candidates.extend([
            (local_dir / "jina-code-embeddings-1.5b", True),
            (local_dir / "jina-embeddings-v2-base-code", True),
            (local_dir / "all-MiniLM-L6-v2", False),
        ])
        remote_candidates.extend([
            ("jinaai/jina-code-embeddings-1.5b", True),
            ("jinaai/jina-embeddings-v2-base-code", True),
            ("all-MiniLM-L6-v2", False),
        ])
    else:
        local_candidates.extend([
            (local_dir / "bge-m3", False),
            (local_dir / "all-MiniLM-L6-v2", False),
        ])
        remote_candidates.extend([
            ("BAAI/bge-m3", False),
            ("all-MiniLM-L6-v2", False),
            ("jinaai/jina-embeddings-v2-base-code", True),
        ])

    for path_obj, trust_remote in local_candidates:
        model_name = str(path_obj)
        if not os.path.isdir(model_name):
            continue
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                trust_remote_code=trust_remote,
            )
            ef(["test"])
            _EMBEDDING_CACHE[profile] = ef
            return ef
        except Exception:
            continue

    for model_name, trust_remote in remote_candidates:
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                trust_remote_code=trust_remote,
            )
            ef(["test"])
            _EMBEDDING_CACHE[profile] = ef
            return ef
        except Exception:
            continue

    ef = embedding_functions.DefaultEmbeddingFunction()
    _EMBEDDING_CACHE[profile] = ef
    return ef


class Retriever:
    def __init__(self, db_path: str = None, collection_name: str = None):
        self._db_path = db_path or config.CHROMA_DB_ROOT
        self._base_collection_name = collection_name or "unity_codebase"
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, domain: str) -> chromadb.Collection:
        collection = self._collections.get(domain)
        if collection is None:
            import os

            os.makedirs(self._db_path, exist_ok=True)
            client = chromadb.PersistentClient(path=self._db_path)
            collection = client.get_or_create_collection(
                name=collection_name_for_domain(self._base_collection_name, domain),
                embedding_function=get_embedding_function(domain),
                metadata={"hnsw:space": "cosine", "domain": domain},
            )
            self._collections[domain] = collection
        return collection

    def reset_collection(self):
        self._collections = {}

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_path: str | None = None,
        filter_type: str | None = None,
        filter_class: str | None = None,
        filter_domain: str | None = None,
    ) -> list[SearchResult]:
        k = top_k or config.RAG_TOP_K
        plan = self._build_query_plan(query, k)
        where = self._build_where(filter_path, filter_type, filter_class)

        if filter_domain:
            domains = [filter_domain]
        elif filter_type:
            domains = list(domains_for_chunk_type(filter_type))
        else:
            domains = list(dict.fromkeys(plan.preferred_domains + plan.secondary_domains))

        candidates = []
        for domain in domains:
            collection = self._get_collection(domain)
            count = collection.count()
            if count == 0:
                continue

            candidate_k = max(k * 2, min(plan.candidate_k, count))
            kwargs = {
                "query_texts": [query],
                "n_results": min(candidate_k, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            raw = collection.query(**kwargs)
            candidates.extend(self._parse_results(raw, domain))

        if not candidates:
            return []

        reranked = self._rerank_results(candidates, plan, k)
        return self._attach_parent_context(reranked)

    def search_by_class(self, class_name: str) -> list[SearchResult]:
        collection = self._get_collection("code")
        raw = collection.get(
            where={"class_name": class_name},
            include=["documents", "metadatas"],
        )
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        results = []
        for document, metadata in zip(documents, metadatas):
            results.append(SearchResult(
                text=document,
                relative_path=metadata.get("relative_path", ""),
                file_path=metadata.get("file_path", ""),
                chunk_type=metadata.get("chunk_type", ""),
                class_name=metadata.get("class_name", ""),
                method_name=metadata.get("method_name", ""),
                namespace=metadata.get("namespace", ""),
                start_line=metadata.get("start_line", 0),
                score=1.0,
                raw_score=1.0,
                rerank_score=1.0,
                collection_domain="code",
                file_ext=metadata.get("file_ext", ""),
                relative_dir=metadata.get("relative_dir", "."),
                top_level_dir=metadata.get("top_level_dir", "."),
                assembly_name=metadata.get("assembly_name", ""),
                asset_kind=metadata.get("asset_kind", ""),
            ))
        return results

    def list_all_classes(self) -> list[dict]:
        collection = self._get_collection("code")
        count = collection.count()
        if count == 0:
            return []

        raw = collection.get(
            where={"chunk_type": "class"},
            include=["metadatas"],
        )
        seen = set()
        classes = []
        for metadata in raw["metadatas"]:
            key = (metadata.get("class_name", ""), metadata.get("relative_path", ""))
            if key not in seen and key[0]:
                seen.add(key)
                classes.append({
                    "class_name": metadata["class_name"],
                    "relative_path": metadata["relative_path"],
                    "namespace": metadata.get("namespace", ""),
                    "assembly_name": metadata.get("assembly_name", ""),
                })
        return sorted(classes, key=lambda item: item["class_name"])

    def list_scenes(self) -> list[dict]:
        collection = self._get_collection("scene")
        count = collection.count()
        if count == 0:
            return []

        raw = collection.get(include=["metadatas"])
        seen = set()
        scenes = []
        for metadata in raw["metadatas"]:
            path = metadata.get("relative_path", "")
            if path and path not in seen:
                seen.add(path)
                scenes.append({"relative_path": path, "file_path": metadata.get("file_path", "")})
        return sorted(scenes, key=lambda item: item["relative_path"])

    def get_index_count(self) -> int:
        total = 0
        for domain in ALL_COLLECTION_DOMAINS:
            try:
                total += self._get_collection(domain).count()
            except Exception:
                continue
        return total

    def get_stats(self) -> dict:
        stats = {
            "total_chunks": 0,
            "db_path": self._db_path,
            "collection_name": self._base_collection_name,
            "type_distribution": {},
            "domain_distribution": {},
            "file_count": 0,
            "embedding_profiles": {
                "code": "jina-code-embeddings-1.5b",
                "scene/prefab/asset/audio/image": "bge-m3",
            },
        }
        file_set: set[str] = set()

        for domain in ALL_COLLECTION_DOMAINS:
            try:
                collection = self._get_collection(domain)
                count = collection.count()
            except Exception:
                continue

            if count == 0:
                continue

            stats["total_chunks"] += count
            stats["domain_distribution"][domain] = count

            try:
                raw = collection.get(limit=count, include=["metadatas"])
            except Exception:
                continue

            for metadata in raw.get("metadatas", []):
                chunk_type = metadata.get("chunk_type", "unknown")
                stats["type_distribution"][chunk_type] = stats["type_distribution"].get(chunk_type, 0) + 1
                if metadata.get("relative_path"):
                    file_set.add(metadata["relative_path"])

        stats["file_count"] = len(file_set)
        return stats

    @staticmethod
    def _build_where(
        filter_path: str | None,
        filter_type: str | None,
        filter_class: str | None,
    ) -> dict | None:
        conditions = []
        if filter_path:
            conditions.append({"relative_path": {"$contains": filter_path}})
        if filter_type:
            conditions.append({"chunk_type": {"$eq": filter_type}})
        if filter_class:
            conditions.append({"class_name": {"$eq": filter_class}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _build_query_plan(query: str, top_k: int) -> QueryPlan:
        lower = query.lower()
        ignored_terms = {
            "asset", "assets", "script", "scripts", "unity", "project",
            "code", "logic", "method", "class", "function", "file", "files",
        }
        audio_keywords = ("wwise", "audio", "sound", "music", "voice", "sfx", "bank", "event", "bus", "rtpc", "switch", "state", "wem", "bnk", "wwu")
        image_keywords = ("image", "sprite", "icon", "texture", "atlas", "portrait", "thumbnail", "background", "ui", "png", "jpg", "jpeg", "psd")
        scene_keywords = ("scene", "hierarchy", "gameobject", ".unity")
        prefab_keywords = ("prefab", "预制体")
        assembly_keywords = ("assembly", "asmdef", "程序集")

        if any(token in lower for token in audio_keywords):
            preferred_domains = ("audio", "code", "asset")
            secondary_domains = ("prefab", "scene", "image")
            preferred_chunk_types = ("audio", "assembly", "class", "method")
            deprioritized_chunk_types = ("scene", "prefab")
            preferred_extensions = (".wwu", ".bnk", ".wem", ".wav", ".ogg", ".mp3")
        elif any(token in lower for token in image_keywords):
            preferred_domains = ("image", "prefab", "scene", "asset", "code")
            secondary_domains = ("audio",)
            preferred_chunk_types = ("image", "prefab", "scene", "asset")
            deprioritized_chunk_types = ("method",)
            preferred_extensions = (".png", ".jpg", ".jpeg", ".tga", ".psd", ".spriteatlas")
        elif any(token in lower for token in scene_keywords):
            preferred_domains = ("scene", "prefab", "code", "asset")
            secondary_domains = ("audio", "image")
            preferred_chunk_types = ("scene", "prefab", "class", "method")
            deprioritized_chunk_types = ("image",)
            preferred_extensions = (".unity", ".prefab", ".cs")
        elif any(token in lower for token in prefab_keywords):
            preferred_domains = ("prefab", "scene", "code", "asset")
            secondary_domains = ("image", "audio")
            preferred_chunk_types = ("prefab", "scene", "class", "method")
            deprioritized_chunk_types = ()
            preferred_extensions = (".prefab", ".unity", ".cs")
        elif any(token in lower for token in assembly_keywords):
            preferred_domains = ("code", "asset")
            secondary_domains = ("scene", "prefab", "audio", "image")
            preferred_chunk_types = ("assembly", "class", "file", "method")
            deprioritized_chunk_types = ("scene", "prefab", "image", "audio")
            preferred_extensions = (".asmdef", ".cs")
        else:
            preferred_domains = ("code", "prefab", "scene", "asset")
            secondary_domains = ("audio", "image")
            preferred_chunk_types = ("method", "class", "file", "assembly")
            deprioritized_chunk_types = ("scene", "prefab", "image")
            preferred_extensions = (".cs", ".asmdef", ".prefab", ".unity")

        path_terms = set()
        symbol_terms = set()
        assembly_terms = set()

        for raw in re.findall(r"[A-Za-z0-9_./\\-]+", query):
            token = raw.strip("./\\-_").replace("\\", "/").lower()
            if len(token) < 2:
                continue
            if "/" in token:
                if len(token) >= 5:
                    path_terms.add(token)
                path_terms.update(
                    part for part in token.split("/")
                    if len(part) >= 2 and part not in ignored_terms
                )
            elif "." in token and any(ch.isalpha() for ch in token):
                symbol_terms.add(token)
                symbol_terms.update(
                    part for part in token.split(".")
                    if len(part) >= 2 and part not in ignored_terms
                )
                if token.endswith(".asmdef"):
                    assembly_terms.add(token[:-7])
            elif len(token) >= 3 and token.endswith((".cs", ".unity", ".prefab", ".asmdef")):
                path_terms.add(token)
                symbol_terms.add(token.rsplit(".", 1)[0])

        for raw in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", query):
            symbol_terms.add(raw.lower())

        for raw in re.findall(r"`([^`]+)`", query):
            cleaned = raw.strip().lower()
            if cleaned:
                symbol_terms.add(cleaned)

        assembly_terms.update(
            term for term in symbol_terms
            if term.endswith("assembly") or term.endswith(".asmdef") or "." in term
        )

        candidate_k = max(top_k * 8, 24)
        return QueryPlan(
            preferred_domains=preferred_domains,
            secondary_domains=secondary_domains,
            preferred_chunk_types=preferred_chunk_types,
            deprioritized_chunk_types=deprioritized_chunk_types,
            preferred_extensions=preferred_extensions,
            path_terms=tuple(sorted(path_terms)),
            symbol_terms=tuple(sorted(term for term in symbol_terms if term not in ignored_terms)),
            assembly_terms=tuple(sorted(assembly_terms)),
            candidate_k=candidate_k,
        )

    def _rerank_results(
        self,
        candidates: list[SearchResult],
        plan: QueryPlan,
        top_k: int,
    ) -> list[SearchResult]:
        for result in candidates:
            result.raw_score = result.score
            result.rerank_score = self._score_result(result, plan)

        remaining = sorted(candidates, key=lambda item: item.rerank_score, reverse=True)
        selected = []
        file_counts: dict[str, int] = {}
        class_counts: dict[tuple[str, str], int] = {}

        while remaining and len(selected) < top_k:
            best_index = 0
            best_score = float("-inf")

            for index, candidate in enumerate(remaining):
                adjusted = candidate.rerank_score
                adjusted -= file_counts.get(candidate.relative_path, 0) * 0.08
                if candidate.class_name:
                    class_key = (candidate.relative_path, candidate.class_name)
                    adjusted -= class_counts.get(class_key, 0) * 0.10
                if adjusted > best_score:
                    best_score = adjusted
                    best_index = index

            chosen = remaining.pop(best_index)
            selected.append(chosen)
            file_counts[chosen.relative_path] = file_counts.get(chosen.relative_path, 0) + 1
            if chosen.class_name:
                class_key = (chosen.relative_path, chosen.class_name)
                class_counts[class_key] = class_counts.get(class_key, 0) + 1

        return selected

    @staticmethod
    def _score_result(result: SearchResult, plan: QueryPlan) -> float:
        score = result.score
        relative_path = result.relative_path.lower()
        relative_dir = result.relative_dir.lower()
        top_level_dir = result.top_level_dir.lower()
        class_name = result.class_name.lower()
        method_name = result.method_name.lower()
        assembly_name = result.assembly_name.lower()
        stem = relative_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        if result.collection_domain in plan.preferred_domains:
            score += 0.18
        if result.collection_domain in plan.secondary_domains:
            score += 0.05
        if result.chunk_type in plan.preferred_chunk_types:
            score += 0.16
        if result.chunk_type in plan.deprioritized_chunk_types:
            score -= 0.12
        if result.file_ext in plan.preferred_extensions:
            score += 0.08

        for path_term in plan.path_terms:
            if path_term in relative_path:
                score += 0.06
            if path_term in relative_dir:
                score += 0.03
            if path_term == top_level_dir:
                score += 0.04
            if path_term == stem:
                score += 0.08

        for symbol_term in plan.symbol_terms:
            if class_name and symbol_term == class_name:
                score += 0.20
            elif class_name and symbol_term in class_name:
                score += 0.08
            if method_name and symbol_term == method_name:
                score += 0.18
            elif method_name and symbol_term in method_name:
                score += 0.07
            if assembly_name and symbol_term in assembly_name:
                score += 0.08
            if symbol_term == stem:
                score += 0.10

        for assembly_term in plan.assembly_terms:
            if assembly_name and assembly_term in assembly_name:
                score += 0.10

        if result.chunk_type == "method":
            score += 0.03
        if result.chunk_type == "class" and result.class_name:
            score += 0.02

        return score

    def _attach_parent_context(self, results: list[SearchResult]) -> list[SearchResult]:
        code_collection = self._get_collection("code")
        cache: dict[tuple[str, str], str] = {}

        for result in results:
            if result.collection_domain != "code" or result.chunk_type != "method" or not result.class_name:
                continue

            cache_key = (result.relative_path, result.class_name)
            if cache_key not in cache:
                parent_context = ""
                raw = code_collection.get(
                    where={"class_name": result.class_name},
                    include=["documents", "metadatas"],
                )
                for document, metadata in zip(raw.get("documents", []), raw.get("metadatas", [])):
                    if (
                        metadata.get("relative_path") == result.relative_path
                        and metadata.get("chunk_type") == "class"
                    ):
                        parent_context = (document or "")[:800]
                        break
                cache[cache_key] = parent_context

            result.parent_context = cache[cache_key]

        return results

    @staticmethod
    def _parse_results(raw: dict, domain: str) -> list[SearchResult]:
        results = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            similarity = 1.0 - distance
            results.append(SearchResult(
                text=document,
                relative_path=metadata.get("relative_path", ""),
                file_path=metadata.get("file_path", ""),
                chunk_type=metadata.get("chunk_type", ""),
                class_name=metadata.get("class_name", ""),
                method_name=metadata.get("method_name", ""),
                namespace=metadata.get("namespace", ""),
                start_line=metadata.get("start_line", 0),
                score=similarity,
                raw_score=similarity,
                rerank_score=similarity,
                collection_domain=domain,
                file_ext=metadata.get("file_ext", ""),
                relative_dir=metadata.get("relative_dir", "."),
                top_level_dir=metadata.get("top_level_dir", "."),
                assembly_name=metadata.get("assembly_name", ""),
                asset_kind=metadata.get("asset_kind", ""),
            ))

        return results


def create_retriever_for_project(project) -> Retriever:
    return Retriever(db_path=project.db_path, collection_name=project.collection_name)
