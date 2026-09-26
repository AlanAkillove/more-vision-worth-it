# FGVC-Aircraft 数据集下载与完整性校验报告

- 生成时间：2026-09-26
- 执行环境：Windows / PowerShell 5.1；curl 8.19.0；bsdtar 3.8.4（Windows 自带 tar.exe）
- 工作区根：`f:\DL Projects\more-vision-worth-it`
- 目标：把完整原始数据集安全下载并解压到 `FGVC_Aircraft\raw\`，并完成完整性校验

---

## 1. 数据来源与下载方式

| 项目 | 值 |
| --- | --- |
| 来源仓库 | `https://www.modelscope.cn/datasets/OmniData/FGVC_Aircraft.git`（ModelScope，git-lfs 跟踪） |
| 实际下载 URL | `https://www.modelscope.cn/api/v1/datasets/OmniData/FGVC_Aircraft/repo?Revision=master&FilePath=raw%2FFGVC_Aircraft.tar.gz.00` |
| 下载方式 | `curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors -o <file> "<url>"`（跟随 302→cdn-lfs-cn-1.modelscope.cn，支持断点续传，无需认证） |
| 落地文件 | `FGVC_Aircraft\raw\FGVC_Aircraft.tar.gz.00` |
| 下载耗时 | 约 **5 分 36 秒**（17:00:55 → 17:06:31），平均约 7.8 MB/s，一次成功未中断 |

> 备注：本次下载未触发断点续传；`-C -` 与 `--retry*` 参数已保留，便于日后重下。

---

## 2. SHA256 与大小校验（下载文件）

| 项目 | 值 |
| --- | --- |
| 实际大小 | `2753776032` 字节 |
| 期望大小 | `2753776032` 字节（LFS pointer 声明） |
| 大小一致 | ✅ 是 |
| 实际 SHA256 | `30DBDBE50524F1E3722324E3D3644F563A0634497175647FA1A2475759F4761E` |
| 期望 SHA256 | `30dbdbe50524f1e3722324e3d3644f563a0634497175647fa1a2475759f4761e`（LFS oid） |
| SHA256 一致 | ✅ 是（严格匹配） |

校验命令：`Get-FileHash -Algorithm SHA256 -Path "<file>"`

**结论：下载文件与 LFS 官方哈希/大小严格一致，下载完整无损。**

---

## 3. 归档结构与解压过程（关键发现：嵌套 tarball）

`tar -tzf` 检查发现，外层 tarball **并非**直接包含 `fgvc-aircraft-2013b\data\`，而是**嵌套双包**结构：

```
外层 FGVC_Aircraft.tar.gz.00  (2,753,776,032 字节)
└── FGVC_Aircraft/
    └── fgvc-aircraft-2013b.tar.gz  (2,753,340,328 字节)   ← 内层包
        └── fgvc-aircraft-2013b/
            ├── data/
            │   ├── images/            (10000 张 *.jpg)
            │   └── *.txt              (标注/划分文件)
            ├── evaluation.m / example_evaluation.m
            ├── README.md / README.html
            └── vl_*.m
```

处理步骤：

1. 解外层：`tar.exe -xzf "raw\FGVC_Aircraft.tar.gz.00" -C "raw"` → 得到 `raw\FGVC_Aircraft\fgvc-aircraft-2013b.tar.gz`（耗时约 2 分 30 秒，exit code 0）
2. 解内层：`tar.exe -xzf "raw\FGVC_Aircraft\fgvc-aircraft-2013b.tar.gz" -C "raw"` → 得到 `raw\fgvc-aircraft-2013b\data\...`（耗时约 2 分 34 秒，exit code 0）

外层与内层 `tar -tzf` 均 exit code 0，gzip 可正常完整读取，无损坏。**两个 tar 包均按要求保留，未删除。**

---

## 4. 最终路径与目录结构

最终数据根目录：

```
f:\DL Projects\more-vision-worth-it\FGVC_Aircraft\raw\
├── FGVC_Aircraft.tar.gz.00                      (2,753,776,032 字节，外层包，保留)
├── FGVC_Aircraft\                               (解外层产生，内含内层包)
│   └── fgvc-aircraft-2013b.tar.gz               (2,753,340,328 字节，内层包，保留)
└── fgvc-aircraft-2013b\                         ← 目标数据根
    └── data\
        ├── images\                              (10000 张 *.jpg)
        ├── variants.txt                         (100 行)
        ├── families.txt                         (70 行)
        ├── manufacturers.txt                    (30 行)
        ├── images_variant_train.txt             (3334 行)
        ├── images_variant_val.txt               (3333 行)
        ├── images_variant_test.txt              (3333 行)
        ├── images_variant_trainval.txt
        ├── images_family_{train,val,test,trainval}.txt
        ├── images_manufacturer_{train,val,test,trainval}.txt
        ├── images_{train,val,test}.txt
        └── images_box.txt
```

- `data\` 目录解压后总大小：**2,762,813,044 字节**（约 2.57 GiB）

---

## 5. 完整性校验逐项结果

| 校验项 | 期望 | 实测 | 结果 |
| --- | --- | --- | --- |
| `data\images\*.jpg` 图像数 | 10000 | **10000** | ✅ |
| `images_variant_train.txt` 行数 | 3334 | **3334** | ✅ |
| `images_variant_val.txt` 行数 | 3333 | **3333** | ✅ |
| `images_variant_test.txt` 行数 | 3333 | **3333** | ✅ |
| `variants.txt` 行数 | 100 | **100** | ✅ |
| `families.txt` 行数 | 70 | **70** | ✅ |
| `manufacturers.txt` 行数 | 30 | **30** | ✅ |
| train 唯一图片 ID 数 | 3334 | **3334** | ✅ |
| val 唯一图片 ID 数 | 3333 | **3333** | ✅ |
| test 唯一图片 ID 数 | 3333 | **3333** | ✅ |
| 重叠 train ∩ val | 0 | **0** | ✅ |
| 重叠 train ∩ test | 0 | **0** | ✅ |
| 重叠 val ∩ test | 0 | **0** | ✅ |
| 三划分并集 | 10000 | **10000** | ✅ |
| 划分中 ID 无对应图像文件数 | 0 | **0** | ✅ |
| 图像文件不在任何划分中的数量 | 0 | **0** | ✅ |
| JPEG 魔数（FF D8 FF）随机抽检 | 全部合格 | **0 / 25 不合格**（25 张全部合格） | ✅ |

**说明：**
- 三个划分文件互不重叠、并集恰为 10000，且与 `images\` 目录中的文件一一对应（双向无遗漏）。
- JPEG 魔数抽检覆盖 25 张随机图片，前 3 字节均为 `FF D8 FF`。
- 行数统计按“非空行”计（文件若含结尾换行不计入）。

---

## 6. 遇到的问题与解决办法

| 问题 | 解决办法 |
| --- | --- |
| 外层归档为**嵌套双 tar.gz**，直接解压只能得到内层 `.tar.gz`，非预期的 `data\` 目录 | 先解外层，再定位并解内层 `FGVC_Aircraft\fgvc-aircraft-2013b.tar.gz` 到 `raw\`，最终得到 `raw\fgvc-aircraft-2013b\data\` |
| 长任务（下载、解压）易受终端阻塞影响 | 全部下载/解压命令写入 `.ps1` 脚本并后台运行，用文件大小与终端输出轮询进度 |
| 含空格路径 + PowerShell 语法 | 路径一律用双引号包裹；语句用 `;` 分隔（未使用 `&&`） |

---

## 7. 遗留风险与建议

1. **游离的 `raw\FGVC_Aircraft\fgvc-aircraft-2013b.tar.gz`（约 2.56 GiB）**：解外层产生的内层包，按“不删除非临时文件”的约束予以保留。若需释放空间，可在确认 `raw\fgvc-aircraft-2013b\data` 无误后删除该内层包（外层包建议一并保留作原始备份）。
2. **仓库内 `sample\image\` 预览图不完整**（仅 99/500 张）：属于预览小图，非本项目所需，本次**未**补全。
3. **`.git\lfs\incomplete\` 残留**：历史 LFS 中断产物，本次通过直连 CDN 下载绕开 git-lfs，未清理该目录（遵守“不做仓库 git 操作”的约束）。
4. **磁盘占用**：当前 `raw\` 下含 外层包 + 内层包 + 解压数据，合计约 8.3 GiB；F 盘剩余约 1.52 TiB，空间充足。
5. **临时文件**：本次过程脚本与清单存放于工作区根 `.fgvc_tmp\`（含 `download_fgvc.ps1`、`extract_outer_list_inner.ps1`、`extract_inner.ps1`、`verify.ps1`、`tar_list.txt`、`inner_list.txt` 等），可按需清理。

---

## 8. 结论

- 下载文件 SHA256 与大小**严格匹配** LFS 官方值，归档完整无损。
- 解压后得到预期结构 `FGVC_Aircraft\raw\fgvc-aircraft-2013b\data\`（images + 标注 txt）。
- 全部完整性校验项**100% 通过**（10000 张图像、3334/3333/3333 划分、互不重叠且并集为 10000、标注文件 100/70/30 行、JPEG 魔数抽检全合格）。
- 数据集**可直接用于训练/评测**。
