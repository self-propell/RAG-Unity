"""
Build and update domain-specific vector indexes for a Unity project.
"""
import hashlib
import json
import os
from pathlib import Path

import chromadb
from rich.console import Console
from rich.progress import track
from rich.table import Table

from core import config
from core.chunker import CodeChunk, chunk_file
from core.projects import get_project_manager
from rag.retriever import (
    ALL_COLLECTION_DOMAINS,
    Retriever,
    collection_name_for_domain,
    domain_for_extension,
    get_embedding_function,
)

console = Console()


def get_collection(
    db_path: str,
    collection_name: str,
    domain: str,
    reset: bool = False,
) -> chromadb.Collection:
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    actual_name = collection_name_for_domain(collection_name, domain)

    if reset:
        try:
            client.delete_collection(actual_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=actual_name,
        embedding_function=get_embedding_function(domain),
        metadata={"hnsw:space": "cosine", "domain": domain},
    )


def reset_project_collections(db_path: str, collection_name: str):
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    for domain in ALL_COLLECTION_DOMAINS:
        try:
            client.delete_collection(collection_name_for_domain(collection_name, domain))
        except Exception:
            continue


def _hash_cache_path(db_path: str) -> str:
    return os.path.join(db_path, "file_hashes.json")


def load_hash_cache(db_path: str) -> dict:
    path = _hash_cache_path(db_path)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hash_cache(cache: dict, db_path: str):
    path = _hash_cache_path(db_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_files(project_path: str) -> list[str]:
    files = []
    for root, dirs, filenames in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in config.IGNORED_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in config.INDEXED_EXTENSIONS:
                files.append(os.path.join(root, filename))
    return sorted(files)


def _load_assembly_index(files: list[str]) -> list[tuple[str, str]]:
    assembly_index = []
    for file_path in files:
        if os.path.splitext(file_path)[1].lower() != ".asmdef":
            continue

        assembly_name = Path(file_path).stem
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            assembly_name = data.get("name") or assembly_name
        except Exception:
            pass

        assembly_index.append((os.path.dirname(file_path), assembly_name))

    assembly_index.sort(key=lambda item: len(item[0]), reverse=True)
    return assembly_index


def _resolve_assembly_name(file_path: str, assembly_index: list[tuple[str, str]]) -> str:
    normalized = os.path.normcase(os.path.abspath(file_path))
    for root_path, assembly_name in assembly_index:
        root_normalized = os.path.normcase(os.path.abspath(root_path))
        if normalized == root_normalized or normalized.startswith(root_normalized + os.sep):
            return assembly_name
    return ""


def _build_file_metadata(
    file_path: str,
    project_root: str,
    assembly_index: list[tuple[str, str]],
) -> dict:
    relative_path = os.path.relpath(file_path, project_root).replace("\\", "/")
    relative_dir = os.path.dirname(relative_path).replace("\\", "/") or "."
    top_level_dir = relative_path.split("/", 1)[0] if "/" in relative_path else "."
    file_ext = os.path.splitext(file_path)[1].lower()

    return {
        "file_ext": file_ext,
        "relative_dir": relative_dir,
        "top_level_dir": top_level_dir,
        "asset_kind": config.INDEXED_EXTENSIONS.get(file_ext, "unknown"),
        "assembly_name": _resolve_assembly_name(file_path, assembly_index),
    }


def _make_chunk_id(chunk: CodeChunk, idx: int) -> str:
    key = f"{chunk.relative_path}::{chunk.chunk_type}::{chunk.class_name}::{chunk.method_name}::{idx}"
    return hashlib.md5(key.encode()).hexdigest()


def index_file(
    collection: chromadb.Collection,
    file_path: str,
    project_root: str,
    assembly_index: list[tuple[str, str]] | None = None,
):
    chunks = chunk_file(file_path, project_root, config.CHUNK_MAX_TOKENS)
    if not chunks:
        return 0

    metadata_extras = _build_file_metadata(file_path, project_root, assembly_index or [])
    for chunk in chunks:
        chunk.metadata.update(metadata_extras)

    try:
        collection.delete(where={"file_path": file_path})
    except Exception:
        pass

    ids = [_make_chunk_id(chunk, idx) for idx, chunk in enumerate(chunks)]
    documents = [chunk.text for chunk in chunks]
    metadatas = [chunk.build_metadata() for chunk in chunks]

    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    return len(chunks)


def _get_domain_collections(db_path: str, collection_name: str, files: list[str]) -> dict[str, chromadb.Collection]:
    domains = {domain_for_extension(os.path.splitext(path)[1].lower()) for path in files}
    return {
        domain: get_collection(db_path, collection_name, domain, reset=False)
        for domain in domains
    }


def build_index(
    project_path: str,
    db_path: str,
    collection_name: str,
    project_id: str = None,
):
    console.print("\n[bold cyan]Building Unity project index[/bold cyan]")
    console.print(f"Project: [green]{project_path}[/green]")
    console.print(f"DB: [dim]{db_path}[/dim]")

    files = scan_files(project_path)
    assembly_index = _load_assembly_index(files)
    console.print(f"Discovered {len(files)} indexable files\n")

    reset_project_collections(db_path, collection_name)
    collections = _get_domain_collections(db_path, collection_name, files)
    hash_cache = {}
    total_chunks = 0
    failed = []

    for file_path in track(files, description="Indexing..."):
        domain = domain_for_extension(os.path.splitext(file_path)[1].lower())
        collection = collections[domain]
        try:
            total_chunks += index_file(collection, file_path, project_path, assembly_index)
            hash_cache[file_path] = file_md5(file_path)
        except Exception as exc:
            failed.append((file_path, str(exc)))

    save_hash_cache(hash_cache, db_path)

    console.print("\n[bold green]Index build complete[/bold green]")
    console.print(f"  Files: {len(files)}")
    console.print(f"  Chunks: {total_chunks}")
    if failed:
        console.print(f"  [yellow]Failed files: {len(failed)}[/yellow]")
        for path, err in failed[:5]:
            console.print(f"    - {os.path.basename(path)}: {err}")

    if project_id:
        get_project_manager().update_index_stats(project_id, total_chunks, len(files))

    return total_chunks, len(files)


def update_index(
    project_path: str,
    db_path: str,
    collection_name: str,
    project_id: str = None,
):
    console.print("\n[bold cyan]Updating Unity project index[/bold cyan]")

    files = scan_files(project_path)
    assembly_index = _load_assembly_index(files)
    hash_cache = load_hash_cache(db_path)
    collections = _get_domain_collections(db_path, collection_name, files)

    changed = []
    deleted = []

    for file_path in files:
        current_hash = file_md5(file_path)
        if hash_cache.get(file_path) != current_hash:
            changed.append(file_path)

    for cached_path in list(hash_cache.keys()):
        if not os.path.exists(cached_path):
            deleted.append(cached_path)

    if not changed and not deleted:
        console.print("[green]Index is already up to date[/green]")
        return

    console.print(f"Changed files: {len(changed)}, deleted files: {len(deleted)}")

    for file_path in deleted:
        domain = domain_for_extension(os.path.splitext(file_path)[1].lower())
        collection = get_collection(db_path, collection_name, domain, reset=False)
        try:
            collection.delete(where={"file_path": file_path})
            hash_cache.pop(file_path, None)
            console.print(f"  [yellow]Removed[/yellow] {os.path.basename(file_path)}")
        except Exception:
            continue

    total_chunks = 0
    for file_path in track(changed, description="Updating..."):
        domain = domain_for_extension(os.path.splitext(file_path)[1].lower())
        collection = collections.setdefault(
            domain,
            get_collection(db_path, collection_name, domain, reset=False),
        )
        try:
            total_chunks += index_file(collection, file_path, project_path, assembly_index)
            hash_cache[file_path] = file_md5(file_path)
        except Exception as exc:
            console.print(f"  [yellow]Skipped[/yellow] {os.path.basename(file_path)}: {exc}")

    save_hash_cache(hash_cache, db_path)
    console.print(f"\n[bold green]Update complete[/bold green], processed {total_chunks} chunks")

    if project_id:
        stats = Retriever(db_path=db_path, collection_name=collection_name).get_stats()
        get_project_manager().update_index_stats(
            project_id,
            stats["total_chunks"],
            stats["file_count"],
        )


def show_stats(db_path: str, collection_name: str):
    retriever = Retriever(db_path=db_path, collection_name=collection_name)
    stats = retriever.get_stats()

    table = Table(title="Index Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("DB Path", stats["db_path"])
    table.add_row("Collection Base", stats["collection_name"])
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Files", str(stats["file_count"]))
    console.print(table)

    if stats["domain_distribution"]:
        domain_table = Table(title="Domain Distribution")
        domain_table.add_column("Domain")
        domain_table.add_column("Chunks")
        for domain, count in sorted(stats["domain_distribution"].items(), key=lambda item: -item[1]):
            domain_table.add_row(domain, str(count))
        console.print(domain_table)

    if stats["type_distribution"]:
        type_table = Table(title="Chunk Type Distribution")
        type_table.add_column("Type")
        type_table.add_column("Count")
        for chunk_type, count in sorted(stats["type_distribution"].items(), key=lambda item: -item[1]):
            type_table.add_row(chunk_type, str(count))
        console.print(type_table)
