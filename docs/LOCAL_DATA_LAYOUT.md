# 本機資料與執行入口

唯一程式來源為本 Repo 的 `src/gpu_pre2` 與 `e2_calibration`。原始量測、固定模型資產、歷史權重在 Repo 外的 `../research_data`；可用 `GPU_PRE2_DATA_ROOT` 指定其他位置。

```text
research_data/
├── calibration/{GB10,RTX3090,RTX4090}/
├── captures/i2/<device>/<workload>/
├── captures/i3/<device>/<stage>/
├── assets/i2_frozen_A/
├── assets/i3_rtx4090_delivery/
├── checkpoints/i4_p0_p2/
├── checkpoints/o3_l4_method_fixture/
└── provenance/
```

`data/manifests/DATASETS.json` 是唯一資料註冊表。程式不掃描舊工作區，不載入歷史外掛，不讀取 `vendor/legacy`，也不把來源快照加入 Python 路徑。未知資料 ID、舊階段、未知 protocol 及越界 symlink 一律拒絕。

`run.sh` 將 PYTHONPATH 設為本 Repo 的 `src` 和根目錄，停用 user site。Python 選擇順序為 `GPU_PRE2_PYTHON`、本機忽略版控的 `local-runtime.json`、`.venv/bin/python`、`python3`。本機沿用已具 PyTorch／NumPy 的環境，沒有安裝、升級 CUDA 或變動服務。其他機器可 `python -m pip install -e .`，收集六模型時另需相容的 TorchVision、Transformers、Ultralytics 和 Nsight Systems。

## 常用命令

從 Repo 根目錄：

```bash
./run.sh -m unittest discover -s tests -v
./run.sh scripts/audit_i3_catalog.py
./run.sh scripts/seal_code.py
```

重播歷史 I4，輸入是修正 stream alias 後的 canonical v2，輸出必須用新的路徑：

```bash
./run.sh -m gpu_pre2.models.historical.predict \
  --input ../research_data/checkpoints/i4_p0_p2/data/predictor_input_v2/repeat2.json \
  --hardware ../research_data/checkpoints/i4_p0_p2/data/predictor_input_v2/hardware.json \
  --models ../research_data/checkpoints/i4_p0_p2/models \
  --out ../runs/my_i4_replay/prediction.json

./run.sh -m gpu_pre2.models.historical.evaluate \
  --dataset-id I4_P0_P2 --out ../runs/my_i4_evaluation
```

Evaluator 重算保存且已封存的三 repeat v2 predictions；它不會重訓模型，也不覆寫歷史 evaluation。實際預測入口不 import evaluator、trainer 或資料準備程式。

```bash
./run.sh scripts/validate_retained.py --out ../runs/new_migration_validation
./run.sh scripts/build_task_coverage.py --out ../runs/new_task_coverage
./run.sh -m gpu_pre2.profiling.replay --stage I2 --cell-index 0 --out ../runs/new_trace_audit
```

`--cell-index` 是 canonical CSV 的零起算資料列；選到 GI 會拒絕，GI 不當成 P trace。驗證程式檢查全部 cells、I4 三 repeat／poison 隔離、所有 I2 P/R，以及 I3 repeat2 全配置 P/R。`--skip-parser` 可只重做資料完整性與 frozen-model 檢查。

## 收集程式與版本

- I2：`python -m gpu_pre2.i2.run_formal --help`
- I3 isolated：`python -m gpu_pre2.i3.isolated.collect --help`
- I3 context：`python -m gpu_pre2.i3.context.collect --help`
- E2：`python -m e2_calibration --help`

收集入口要求明確的 asset manifest，核對 bytes／hash；沒有「尋找最近一次舊資料」的 fallback。I3 context 依 A／BC manifest 的固定 plan SHA 選擇對應 protocol，未知 plan 拒絕。每個新 run 固定程式 hash，不複製整套 source。新 port protocol 加 `gpu_pre2:` 前綴；I3 使用 source manifest 替代 source snapshot，計時參數保留 40×3。此 packaging 改變已版本化，未宣稱通過三設備新 GPU smoke。

離線 parser 只允許已註冊的 I2/I3 協定；4090 Groups4 的四 stream exception 限於原配置及原 conditional protocol。新匯出的四 stream aliases 會保留實際區別；舊 v1 predictor exports 僅是歷史產物，不能作目前 I4 輸入。

## 證據保存

每份 raw 的 bytes 與原始 metadata 不改寫。Metadata 裡的舊絕對路徑保留為歷史紀錄；有效載入只用新註冊表和相對路徑。

重複 source snapshot 收斂為 `provenance/source_blobs/<sha256>`，原路徑至內容對照在 `SOURCE_BLOBS.jsonl`。這些是無副檔名的來源證據，不可作執行路徑。原始 I4 fit 的 v1 input 留作 frozen training provenance，預測入口使用 v2。原 shell 重現腳本亦已轉成 inert evidence，請使用上方的新命令。

完整逐檔索引在 `provenance/ASSET_INDEX.jsonl`，其 SHA、檔案數及大小由 Git 內的 `data/manifests/ASSET_INDEX.json` 指向。新 GitHub 不包含 raw captures、固定 tensors 或權重。

Strict reference 仍未驗證。3090 isolated operand binaries 與 4090 部分 smoke 交付缺口、軟體 stack 差異、Memset／copy、multistream GI completion 及 S4 單一 context group 等限制，均未因遷移而解除。
