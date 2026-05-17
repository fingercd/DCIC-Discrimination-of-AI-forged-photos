# 延迟运行（Sleep）修改记录

本文档记录为「生成 CSV」与「rerun 重跑」脚本在代码最开始增加固定时长 sleep 的所有修改。若后续因报错再次修改，将追加到本记录中。

---

## 修改日期

2025-03-14（按实际修改日期填写）

---

## 一、generate_csv.py（生成 CSV）

**需求**：脚本在三个小时后才真正开始生成 CSV，即在代码最开始 sleep 3 小时。

**修改内容**：

1. **文件**：`scripts/generate_csv.py`

2. **新增 import**：
   - 增加 `import time`，用于 `time.sleep()`。

3. **在 `main()` 开头增加 3 小时等待**：
   - 在 `def main():` 下、任何参数解析与业务逻辑之前增加：
     - `delay_seconds = 3 * 3600`
     - 打印：`[Delay] 等待 3.0 小时后生成 CSV…`
     - `time.sleep(delay_seconds)`
     - 打印：`[Delay] 等待结束，开始生成 CSV。`
   - 之后再执行原有的 `argparse` 与读写 JSON/CSV 逻辑。

4. **文档注释**：
   - 在文件顶部 docstring 中补充说明：运行后先等待 3 小时再执行生成逻辑。

**结果**：运行 `python scripts/generate_csv.py` 后，进程会先挂起 3 小时，再读取 JSON 并写入 CSV。若运行时报错，请将报错信息与后续修复一并追加到本文档。

---

## 二、rerun_bad_inference.py（重跑问题图）

**需求**：脚本在四个小时后才真正开始重跑推理与写回 JSON，即在代码最开始 sleep 4 小时。

**修改内容**：

1. **文件**：`scripts/rerun_bad_inference.py`

2. **已有 import**：
   - 已有 `import time`，无需新增。

3. **在 `main()` 开头增加 4 小时等待**：
   - 在参数解析之后、`output_path` 与 `test_dir` 等逻辑之前（即“真正开始读 JSON、算 rerun 集合”之前）增加：
     - `delay_seconds = 4 * 3600`
     - 打印：`[Delay] 等待 4.0 小时后开始重跑推理…`
     - `time.sleep(delay_seconds)`
     - 打印：`[Delay] 等待结束，开始执行。`
   - 之后保持原有流程：读 JSON、确定重跑集合、VLM 推理、写回等。

4. **文档注释**：
   - 在文件顶部 docstring 中补充说明：运行后先等待 4 小时再执行推理与写回。

**结果**：运行 `python scripts/rerun_bad_inference.py`（及所需参数）后，进程会先挂起 4 小时，再执行重跑与覆盖。若运行时报错，请将报错信息与后续修复一并追加到本文档。

---

## 三、inference.py 每 10 张保存修复（2025-03-14）

**问题**：原逻辑先跑完所有 VLM（200+ 张）再按 chunk 跑 Unet 并保存，导致在 VLM 阶段长时间无写入，进程中断会丢失内存中的结果。

**修改**：
- 改为按 chunk（每 10 张）处理：每批先跑 VLM → 再跑 Unet/SAM2 → 立即 `flush_results` 写入 JSON。
- `run_vlm_inference` 增加 `reuse_model`、`reuse_processor` 参数，支持跨 chunk 复用模型，避免每批重复加载。
- Plan A 与 Plan B 均按此逻辑执行，确保每处理 10 张就保存一次。

---

## 四、后续报错与修复（若有）

若在实际运行中发生报错并进行了修改，请按以下格式追加到本小节：

- **脚本名**：
- **报错信息**：（粘贴或简述）
- **修复方式**：（说明修改了哪一文件、哪几行、做了什么改动）
- **修复日期**：

---

## 五、train_vlm.py（VLM 训练）

**需求**：脚本在 6.5 小时后再开始训练，即在代码最开始 sleep 6.5 小时。

**修改内容**：

1. **文件**：`scripts/train_vlm.py`
2. **新增 import**：`import time`
3. **在 `main()` 开头增加 6.5 小时等待**：
   - `delay_seconds = 6.5 * 3600`
   - 打印：`[Delay] 等待 6.5 小时后开始训练…`
   - `time.sleep(delay_seconds)`
   - 打印：`[Delay] 等待结束，开始训练。`

---

## 六、如何取消延迟（恢复立即执行）

- **generate_csv.py**：删除或注释掉 `main()` 开头的 `delay_seconds`、`print` 与 `time.sleep(delay_seconds)` 三处，保留其余逻辑即可立即执行。
- **rerun_bad_inference.py**：同上，删除或注释掉 `main()` 中新增的 4 小时 `delay_seconds`、`print` 与 `time.sleep(delay_seconds)` 即可立即执行。
- **train_vlm.py**：同上，删除或注释掉 `main()` 中新增的 6.5 小时 sleep 相关代码即可立即执行。
