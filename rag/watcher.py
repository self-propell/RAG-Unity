"""
Watch Unity project files and keep the domain-specific index up to date.
"""
import os
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from rich.console import Console

from core import config
from rag.indexer import (
    _load_assembly_index,
    file_md5,
    get_collection,
    index_file,
    load_hash_cache,
    save_hash_cache,
    scan_files,
)
from rag.retriever import domain_for_extension

console = Console()


class UnityFileHandler(FileSystemEventHandler):
    def __init__(self, project_path: str, db_path: str, collection_name: str):
        self._project_path = project_path
        self._db_path = db_path
        self._collection_name = collection_name
        self.hash_cache = load_hash_cache(db_path)
        self._pending: set[str] = set()

    def _should_index(self, path: str) -> bool:
        return Path(path).suffix.lower() in config.INDEXED_EXTENSIONS

    def _get_assembly_index(self) -> list[tuple[str, str]]:
        return _load_assembly_index(scan_files(self._project_path))

    def on_modified(self, event):
        if not event.is_directory and self._should_index(event.src_path):
            self._handle_change(event.src_path, "updated")

    def on_created(self, event):
        if not event.is_directory and self._should_index(event.src_path):
            self._handle_change(event.src_path, "created")

    def on_deleted(self, event):
        if not event.is_directory and self._should_index(event.src_path):
            self._handle_delete(event.src_path)

    def _handle_change(self, file_path: str, action: str):
        if file_path in self._pending:
            return
        self._pending.add(file_path)

        try:
            time.sleep(0.5)
            domain = domain_for_extension(Path(file_path).suffix.lower())
            collection = get_collection(self._db_path, self._collection_name, domain, reset=False)
            chunks = index_file(collection, file_path, self._project_path, self._get_assembly_index())
            self.hash_cache[file_path] = file_md5(file_path)
            save_hash_cache(self.hash_cache, self._db_path)
            rel = Path(file_path).relative_to(self._project_path)
            console.print(f"[green]{action}[/green] {rel} -> {chunks} chunks ({domain})")
        except Exception as exc:
            console.print(f"[red]index update failed[/red] {file_path}: {exc}")
        finally:
            self._pending.discard(file_path)

    def _handle_delete(self, file_path: str):
        try:
            domain = domain_for_extension(Path(file_path).suffix.lower())
            collection = get_collection(self._db_path, self._collection_name, domain, reset=False)
            collection.delete(where={"file_path": file_path})
            self.hash_cache.pop(file_path, None)
            save_hash_cache(self.hash_cache, self._db_path)
            console.print(f"[yellow]removed[/yellow] {Path(file_path).name} ({domain})")
        except Exception as exc:
            console.print(f"[red]delete failed[/red]: {exc}")


def start_watcher(project_path: str, db_path: str, collection_name: str, project_name: str = ""):
    display_name = project_name or project_path
    console.print("[bold cyan]File watcher started[/bold cyan]")
    console.print(f"Project: [green]{display_name}[/green]")
    console.print(f"Path: [dim]{project_path}[/dim]")
    console.print("Changes will update the domain-specific index automatically. Press Ctrl+C to stop.\n")

    handler = UnityFileHandler(project_path, db_path, collection_name)
    observer = Observer()
    observer.schedule(handler, project_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Watcher stopped[/yellow]")

    observer.join()
