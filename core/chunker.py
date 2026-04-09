"""
Split Unity project files into retrieval-friendly chunks.

Strategy:
1. C# files are chunked by class and member boundaries when tree-sitter is available.
2. Unity YAML assets are summarized by logical object sections instead of raw file heads.
3. Wwise and binary/media assets are indexed with metadata-first descriptors.
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

try:
    import tree_sitter_c_sharp as tscsharp
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
    CS_LANGUAGE = Language(tscsharp.language())
    _parser = Parser(CS_LANGUAGE)
except Exception:
    TREE_SITTER_AVAILABLE = False
    _parser = None


UNITY_YAML_EXTENSIONS = {".unity", ".prefab", ".asset", ".spriteatlas"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tga", ".psd", ".exr"}
AUDIO_NAME_ONLY_EXTENSIONS = {".bnk", ".wem", ".wav", ".ogg", ".mp3"}
KEY_YAML_FIELDS = (
    "m_Name:",
    "m_TagString:",
    "m_Layer:",
    "m_Script:",
    "m_IsActive:",
    "m_Enabled:",
    "m_Sprite:",
    "m_Material:",
    "m_Mesh:",
    "m_Mixer:",
    "m_EditorClassIdentifier:",
)


@dataclass
class CodeChunk:
    text: str
    file_path: str
    relative_path: str
    chunk_type: str
    class_name: str = ""
    method_name: str = ""
    namespace: str = ""
    start_line: int = 0
    end_line: int = 0
    metadata: dict = field(default_factory=dict)

    def build_metadata(self) -> dict:
        metadata = {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "chunk_type": self.chunk_type,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "namespace": self.namespace,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        metadata.update(self.metadata)
        return metadata


def chunk_file(file_path: str, project_root: str, max_tokens: int = 600) -> list[CodeChunk]:
    ext = os.path.splitext(file_path)[1].lower()
    relative_path = os.path.relpath(file_path, project_root).replace("\\", "/")

    if ext in IMAGE_EXTENSIONS:
        return [_chunk_named_asset(file_path, relative_path, "image")]
    if ext in AUDIO_NAME_ONLY_EXTENSIONS:
        return [_chunk_named_asset(file_path, relative_path, "audio")]

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return []

    if not content.strip():
        return []

    if ext == ".cs":
        if TREE_SITTER_AVAILABLE:
            chunks = _chunk_cs_treesitter(content, file_path, relative_path)
        else:
            chunks = _chunk_cs_regex(content, file_path, relative_path)

        result = []
        for chunk in chunks:
            result.extend(_split_if_too_long(chunk, max_tokens))
        return result

    if ext == ".asmdef":
        return _chunk_asmdef(content, file_path, relative_path)

    if ext in UNITY_YAML_EXTENSIONS:
        return _chunk_unity_yaml(content, file_path, relative_path, ext)

    if ext == ".wwu":
        return _chunk_wwise_workunit(content, file_path, relative_path)

    chunk_type = {
        ".unity": "scene",
        ".prefab": "prefab",
        ".asmdef": "assembly",
        ".wwu": "audio",
    }.get(ext, "asset")
    return [_chunk_text_summary(content, file_path, relative_path, chunk_type)]


def _chunk_cs_treesitter(content: str, file_path: str, relative_path: str) -> list[CodeChunk]:
    source = content.encode("utf-8")
    tree = _parser.parse(source)
    root = tree.root_node
    chunks = []

    namespace = _extract_namespace(root, source)

    for class_node in _find_nodes(root, {"class_declaration", "struct_declaration", "interface_declaration", "enum_declaration"}):
        class_name = _node_field_text(class_node, "name", source)
        class_lines = (class_node.start_point[0] + 1, class_node.end_point[0] + 1)

        class_header = _extract_class_header(class_node, source)
        header_chunk = CodeChunk(
            text=class_header,
            file_path=file_path,
            relative_path=relative_path,
            chunk_type="class",
            class_name=class_name,
            namespace=namespace,
            start_line=class_lines[0],
            end_line=class_lines[1],
        )
        header_chunk.metadata = header_chunk.build_metadata()
        chunks.append(header_chunk)

        for method_node in _find_nodes(class_node, {
            "method_declaration",
            "constructor_declaration",
            "property_declaration",
            "field_declaration",
            "event_declaration",
            "indexer_declaration",
        }):
            method_name = _node_field_text(method_node, "name", source)
            method_text = source[method_node.start_byte:method_node.end_byte].decode("utf-8", errors="ignore")
            method_lines = (method_node.start_point[0] + 1, method_node.end_point[0] + 1)

            member_chunk = CodeChunk(
                text=method_text,
                file_path=file_path,
                relative_path=relative_path,
                chunk_type="method",
                class_name=class_name,
                method_name=method_name,
                namespace=namespace,
                start_line=method_lines[0],
                end_line=method_lines[1],
            )
            member_chunk.metadata = member_chunk.build_metadata()
            chunks.append(member_chunk)

    if not chunks:
        fallback = CodeChunk(
            text=content,
            file_path=file_path,
            relative_path=relative_path,
            chunk_type="file",
            start_line=1,
            end_line=content.count("\n") + 1,
        )
        fallback.metadata = fallback.build_metadata()
        chunks.append(fallback)

    return chunks


def _find_nodes(node, target_types: set):
    results = []
    queue = list(node.children)
    while queue:
        current = queue.pop(0)
        if current.type in target_types:
            results.append(current)
        else:
            queue.extend(current.children)
    return results


def _node_field_text(node, field_name: str, source: bytes) -> str:
    child = node.child_by_field_name(field_name)
    if child:
        return source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
    return ""


def _extract_namespace(root, source: bytes) -> str:
    for child in root.children:
        if child.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
            name_node = child.child_by_field_name("name")
            if name_node:
                return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
    return ""


def _extract_class_header(class_node, source: bytes) -> str:
    lines = []
    for child in class_node.children:
        if child.type in ("method_declaration", "constructor_declaration"):
            continue
        text = source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
        lines.append(text)
    return "\n".join(lines)[:2000]


_CLASS_RE = re.compile(
    r"((?:(?:public|private|protected|internal|static|abstract|sealed|partial)\s+)*"
    r"(?:class|struct|interface|enum)\s+\w+[^{]*\{)",
    re.MULTILINE,
)


def _chunk_cs_regex(content: str, file_path: str, relative_path: str) -> list[CodeChunk]:
    chunks = []
    lines = content.split("\n")

    ns_match = re.search(r"namespace\s+([\w.]+)", content)
    namespace = ns_match.group(1) if ns_match else ""

    class_starts = []
    for index, line in enumerate(lines):
        if _CLASS_RE.search(line):
            match = re.search(r"(?:class|struct|interface|enum)\s+(\w+)", line)
            class_name = match.group(1) if match else f"Unknown_{index}"
            class_starts.append((index, class_name))

    if not class_starts:
        fallback = CodeChunk(
            text=content,
            file_path=file_path,
            relative_path=relative_path,
            chunk_type="file",
            namespace=namespace,
            start_line=1,
            end_line=len(lines),
        )
        fallback.metadata = fallback.build_metadata()
        return [fallback]

    for idx, (start_line, class_name) in enumerate(class_starts):
        end_line = class_starts[idx + 1][0] if idx + 1 < len(class_starts) else len(lines)
        class_text = "\n".join(lines[start_line:end_line])
        chunk = CodeChunk(
            text=class_text,
            file_path=file_path,
            relative_path=relative_path,
            chunk_type="class",
            class_name=class_name,
            namespace=namespace,
            start_line=start_line + 1,
            end_line=end_line,
        )
        chunk.metadata = chunk.build_metadata()
        chunks.append(chunk)

    return chunks


def _chunk_unity_yaml(content: str, file_path: str, relative_path: str, ext: str) -> list[CodeChunk]:
    chunk_type = {".unity": "scene", ".prefab": "prefab"}.get(ext, "asset")
    sections = re.split(r"(?m)^--- !u!", content)
    summaries: list[tuple[str, int, int]] = []
    current_line = 1

    for section in sections:
        line_count = section.count("\n") + 1
        stripped = section.strip()
        if stripped:
            summary = _summarize_yaml_section(stripped, chunk_type, relative_path)
            if summary:
                summaries.append((summary, current_line, current_line + line_count - 1))
        current_line += line_count

    if not summaries:
        return [_chunk_text_summary(content, file_path, relative_path, chunk_type)]

    return _pack_summaries(summaries, file_path, relative_path, chunk_type)


def _summarize_yaml_section(section: str, chunk_type: str, relative_path: str) -> str:
    lines = [line.rstrip() for line in section.splitlines() if line.strip()]
    if not lines:
        return ""

    header = lines[0]
    type_match = re.match(r"(\d+)\s+&", header)
    object_type = type_match.group(1) if type_match else "unknown"

    name = ""
    key_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("m_Name:"):
            name = stripped.split(":", 1)[1].strip()
        if any(stripped.startswith(prefix) for prefix in KEY_YAML_FIELDS):
            key_lines.append(stripped)

    if not name and not key_lines:
        return ""

    label = name or f"{chunk_type}_object"
    body = [
        f"AssetType: {chunk_type}",
        f"Path: {relative_path}",
        f"ObjectTypeId: {object_type}",
        f"ObjectName: {label}",
    ]
    if key_lines:
        body.append("KeyFields:")
        body.extend(key_lines[:12])
    return "\n".join(body)[:1600]


def _chunk_asmdef(content: str, file_path: str, relative_path: str) -> list[CodeChunk]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [_chunk_text_summary(content, file_path, relative_path, "assembly")]

    assembly_name = data.get("name") or os.path.splitext(os.path.basename(file_path))[0]
    references = ", ".join(data.get("references", [])[:20]) or "(none)"
    include_platforms = ", ".join(data.get("includePlatforms", [])[:10]) or "(all)"
    exclude_platforms = ", ".join(data.get("excludePlatforms", [])[:10]) or "(none)"
    define_constraints = ", ".join(data.get("defineConstraints", [])[:10]) or "(none)"

    text = "\n".join([
        "Assembly Definition",
        f"Path: {relative_path}",
        f"Name: {assembly_name}",
        f"Root Namespace: {data.get('rootNamespace', '') or '(default)'}",
        f"References: {references}",
        f"Include Platforms: {include_platforms}",
        f"Exclude Platforms: {exclude_platforms}",
        f"Define Constraints: {define_constraints}",
        f"Allow Unsafe Code: {bool(data.get('allowUnsafeCode', False))}",
        f"Auto Referenced: {bool(data.get('autoReferenced', True))}",
    ])
    chunk = CodeChunk(
        text=text,
        file_path=file_path,
        relative_path=relative_path,
        chunk_type="assembly",
        class_name=assembly_name,
        start_line=1,
        end_line=content.count("\n") + 1,
    )
    chunk.metadata = chunk.build_metadata()
    return [chunk]


def _chunk_wwise_workunit(content: str, file_path: str, relative_path: str) -> list[CodeChunk]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return [_chunk_text_summary(content, file_path, relative_path, "audio")]

    grouped: dict[str, list[str]] = {}
    for element in root.iter():
        tag = element.tag.split("}", 1)[-1]
        name = (
            element.attrib.get("Name")
            or element.attrib.get("ObjectName")
            or element.attrib.get("ShortName")
        )
        if not name:
            continue
        if tag.lower() in {"root", "childrenlist"}:
            continue
        grouped.setdefault(tag, [])
        if name not in grouped[tag]:
            grouped[tag].append(name)

    if not grouped:
        return [_chunk_named_asset(file_path, relative_path, "audio")]

    summaries = []
    for tag, names in sorted(grouped.items()):
        lines = [
            "Wwise Work Unit",
            f"Path: {relative_path}",
            f"NodeType: {tag}",
            "Entries:",
        ]
        lines.extend(f"- {name}" for name in names[:80])
        summaries.append(("\n".join(lines)[:1800], 1, content.count("\n") + 1))

    return _pack_summaries(summaries, file_path, relative_path, "audio")


def _chunk_named_asset(file_path: str, relative_path: str, chunk_type: str) -> CodeChunk:
    stem = os.path.splitext(os.path.basename(relative_path))[0]
    tokens = _tokenize_name(stem)
    path_tokens = [part for part in re.split(r"[\\/._-]+", relative_path) if part]

    text = "\n".join([
        f"AssetType: {chunk_type}",
        f"Path: {relative_path}",
        f"FileName: {os.path.basename(relative_path)}",
        f"Stem: {stem}",
        f"NameTokens: {', '.join(tokens) or '(none)'}",
        f"PathTokens: {', '.join(path_tokens[:20])}",
    ])
    chunk = CodeChunk(
        text=text,
        file_path=file_path,
        relative_path=relative_path,
        chunk_type=chunk_type,
        class_name=stem,
        start_line=1,
        end_line=1,
    )
    chunk.metadata = chunk.build_metadata()
    return chunk


def _chunk_text_summary(content: str, file_path: str, relative_path: str, chunk_type: str) -> CodeChunk:
    summary = content[:3000] + ("\n...[truncated]" if len(content) > 3000 else "")
    chunk = CodeChunk(
        text=summary,
        file_path=file_path,
        relative_path=relative_path,
        chunk_type=chunk_type,
        start_line=1,
        end_line=content.count("\n") + 1,
    )
    chunk.metadata = chunk.build_metadata()
    return chunk


def _pack_summaries(
    summaries: list[tuple[str, int, int]],
    file_path: str,
    relative_path: str,
    chunk_type: str,
    max_chars: int = 1800,
) -> list[CodeChunk]:
    chunks = []
    buffer = []
    buffer_start = summaries[0][1]
    buffer_end = summaries[0][2]
    char_count = 0

    for text, start_line, end_line in summaries:
        if buffer and char_count + len(text) + 2 > max_chars:
            chunk = CodeChunk(
                text="\n\n".join(buffer),
                file_path=file_path,
                relative_path=relative_path,
                chunk_type=chunk_type,
                start_line=buffer_start,
                end_line=buffer_end,
            )
            chunk.metadata = chunk.build_metadata()
            chunks.append(chunk)
            buffer = []
            char_count = 0
            buffer_start = start_line

        if not buffer:
            buffer_start = start_line
        buffer.append(text)
        buffer_end = end_line
        char_count += len(text) + 2

    if buffer:
        chunk = CodeChunk(
            text="\n\n".join(buffer),
            file_path=file_path,
            relative_path=relative_path,
            chunk_type=chunk_type,
            start_line=buffer_start,
            end_line=buffer_end,
        )
        chunk.metadata = chunk.build_metadata()
        chunks.append(chunk)

    return chunks


def _tokenize_name(name: str) -> list[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    normalized = re.sub(r"[_\-.]+", " ", normalized)
    return [token.lower() for token in normalized.split() if len(token) >= 2]


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _split_if_too_long(chunk: CodeChunk, max_tokens: int) -> list[CodeChunk]:
    if _estimate_tokens(chunk.text) <= max_tokens:
        return [chunk]

    lines = chunk.text.split("\n")
    avg_chars_per_line = max(1, len(chunk.text) // max(1, len(lines)))
    target_lines = max(10, (max_tokens * 4) // avg_chars_per_line)

    result = []
    for index in range(0, len(lines), target_lines):
        part_lines = lines[index:index + target_lines]
        part = CodeChunk(
            text="\n".join(part_lines),
            file_path=chunk.file_path,
            relative_path=chunk.relative_path,
            chunk_type=chunk.chunk_type,
            class_name=chunk.class_name,
            method_name=chunk.method_name,
            namespace=chunk.namespace,
            start_line=chunk.start_line + index,
            end_line=chunk.start_line + index + len(part_lines),
            metadata=dict(chunk.metadata),
        )
        part.metadata = part.build_metadata()
        result.append(part)
    return result
