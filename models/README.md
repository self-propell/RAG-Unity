# Embedding Models

本目录存放本地 Embedding 模型，需要手动下载。

## 必需模型

### 1. jina-embeddings-v2-base-code (2.9G)
- **用途**: C# 代码专用 Embedding
- **下载**:
  ```bash
  pip install huggingface-hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v2-base-code', local_dir='models/jina-code-embeddings-1.5b')"
  ```
- **HuggingFace**: https://huggingface.co/jinaai/jina-embeddings-v2-base-code

### 2. bge-m3 (4.3G)
- **用途**: 多语言通用 Embedding（Scene/Prefab/Asset/Audio/Image）
- **下载**:
  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
  ```
- **HuggingFace**: https://huggingface.co/BAAI/bge-m3

## 可选模型

### 3. all-MiniLM-L6-v2 (88M)
- **用途**: 轻量级通用 Embedding（可选）
- **下载**:
  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')"
  ```
- **HuggingFace**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## 国内加速

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 然后执行上面的下载命令
```

## 目录结构

下载完成后，目录结构应该是：

```
models/
├── README.md (本文件)
├── jina-code-embeddings-1.5b/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── bge-m3/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── all-MiniLM-L6-v2/ (可选)
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

## 注意事项

- 这些模型文件很大（总共约 7GB），不会提交到 Git
- 每个开发者需要自行下载到本地
- 首次下载可能需要较长时间，请耐心等待
- 下载完成后即可离线使用，不再需要网络连接
