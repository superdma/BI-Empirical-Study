# 把报告发布成网站

本站用 **MkDocs + Material for MkDocs** 构建：源文件还是 `docs/*.md`，构建产物是
纯静态的 `site/`，可以放在任何静态托管上，不需要数据库或后端。

## 1. 本地构建与预览

```bash
cd /media/dut-aiid-vision/storage4/chephyresearch/vllm/bi_validation

# 首次：创建独立的 Python 环境（不影响系统 Python）
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python -r requirements-docs.txt

# 构建 + 检查站内链接
bash scripts/build_site.sh

# 本地预览（自动重载）：http://127.0.0.1:8001
bash scripts/serve_site.sh
```

构建完成后，`site/` 就是完整网站，可以直接打包：

```bash
tar czf bi_site.tar.gz -C site .
```

## 2. 发布前已经处理的隐私问题

- 文档源里的家目录统一写成 `$HOME`，不包含具体账号；
- 脚本用 `${HOME}` 或 `os.path.expanduser` 取模型和环境路径；
- 网站只构建 `docs/`，`results/`、`*.py`、`*.sh` 不会进入 `site/`；
- `results/` 中的 JSON 已去掉家目录前缀，原始日志未纳入仓库；
- `docs/` 中没有服务器 IP、密码或 SSH 信息。

发布前仍建议复查一次：

```bash
grep -rn "PRIVATE KEY" docs/ || echo "clean"
```

## 3. 托管方案

### 方案 A：GitHub Pages（推荐，公开、免费）

适合“给别人一个链接就能看”的场景。

1. 在 GitHub 新建仓库，把本目录（至少 `docs/`、`mkdocs.yml`、
   `requirements-docs.txt`、`scripts/`、`.github/`）推上去；
2. 仓库 **Settings → Pages → Build and deployment → Source** 选
   **GitHub Actions**；
3. 推送到 `main` 后，`.github/workflows/pages.yml` 会自动构建并发布；
4. 站点地址形如 `https://<user>.github.io/<repo>/`。

如果第一次 workflow 在 `configure-pages` 步骤失败，通常是 Pages 还没启用。
完成第 2 步后，在 Actions 页面点 **Re-run jobs** 即可。

> 注意：`mkdocs.yml` 里的 `site_url` 目前没写。绑定自定义域名或使用项目页时，
> 建议补上 `site_url`，否则 sitemap 里的绝对链接不完整（不影响阅读）。

### 方案 B：Cloudflare Pages（全球 CDN，免费）

1. 把仓库推到 GitHub/GitLab；
2. Cloudflare Dashboard → Workers & Pages → Create → Pages → Connect to Git；
3. 构建命令：`pip install -r requirements-docs.txt && mkdocs build`；
4. 输出目录：`site`；
5. 之后每次 push 自动部署，也可以绑定自定义域名。

如果不想连仓库，也可以本地 `bash scripts/build_site.sh` 后，把 `site/` 目录
直接拖到 Cloudflare Pages 的“Direct Upload”。

### 方案 C：实验室/内网自托管

只给内网同事看时，最省事的是静态文件服务器：

```bash
cd site
python3 -m http.server 8080 --bind 0.0.0.0
# 浏览器访问 http://<内网IP>:8080/
```

需要长期运行时，用 nginx：

```nginx
server {
    listen 80;
    server_name _;
    root /path/to/bi_validation/site;
    index index.html;

    location / {
        try_files $uri $uri/ $uri.html =404;
    }
}
```

把 `site/` 同步到 nginx 的 `root` 目录即可；更新时重新构建并覆盖。

### 方案 D：Netlify Drop（最快的一次性分享）

打开 Netlify Drop，把 `site/` 文件夹拖进去，几秒后得到一个临时网址。适合
评审、临时演示；长期使用仍建议接 Git。

## 4. 日常维护

- 改完 `docs/*.md` 后重新运行 `bash scripts/build_site.sh`；
- 新增页面时，在 `mkdocs.yml` 的 `nav` 里加一行，否则页面能构建但不在导航中；
- 文档之间的链接继续用相对 `.md` 路径（如 `06_boundary_results.md`），
  MkDocs 会自动转换成站点 URL；
- 升级 MkDocs Material 后重新构建，并跑一次站内链接检查。

## 5. 常用命令速查

| 目的 | 命令 |
|---|---|
| 安装依赖 | `uv pip install --python .venv-docs/bin/python -r requirements-docs.txt` |
| 构建 | `bash scripts/build_site.sh` |
| 预览 | `bash scripts/serve_site.sh` |
| 检查链接 | `python3 scripts/check_site_links.py site` |
| 打包 | `tar czf bi_site.tar.gz -C site .` |
