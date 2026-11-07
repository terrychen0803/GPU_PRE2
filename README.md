# GPU_PRE2 — RTX4090 → RTX3090 / GB10 Runtime Prediction

以 RTX4090 **P-only 數值事件**預測 RTX3090／GB10 的 operator P CTX 與完整單次 forward GI。六模型：ResNet-50、MobileNetV3-Large、Swin-T、U-Net、YOLO26n、Qwen3-0.6B；Qwen prefill／decode 分開，共七 workload groups。

本機 E2／I2／I3／I4 資產已整合，舊 E1/E3/E4、O1/O2、O3 大量歷史消融與重複工作樹已清理。程式只從本 Repo 載入；資料從明確註冊的 `GPU_PRE2_DATA_ROOT` 讀取，不會載入舊目錄或 legacy 外掛。

## 可用內容

| 階段 | 程式 | 資料／狀態 |
| --- | --- | --- |
| E2 | `e2_calibration/` | 三設備歷史校正、profile、樣本與相容性紀錄 |
| I2 | `gpu_pre2.i2` | 七 A adapters、GI/P/R worker、coordinator、固定資產驗證；189 cells／7,560 measured |
| I3 | `gpu_pre2.i3.isolated`、`.context` | S1～S3、S4 A/BC 共用程式；3,024 cells／120,960 measured |
| Profiling | `gpu_pre2.profiling` | 明確 protocol gate、SQLite 3.28/3.29/3.31、P/R 對齊與品質解析 |
| I4 歷史基準 | `gpu_pre2.models.historical` | 精確候選片段 E1 27 維、六組 frozen models、三 repeat v2 預測與評估重播 |
| I4 新實驗 | `gpu_pre2.models.train_mixed` | 既有 grouped isolated/context mixed trainer；mean-pooling 特徵與歷史基準不同，權重不互換 |
| Task Coverage | `results/coverage/` | 7,200 列實際 I2 R/P 對照；ATen＋shape match，不宣稱完整語意 exact coverage |
| I5 | `docs/DATA_CONTRACT.md` | GI 契約；完整 iteration predictor 尚未實作 |

**目前仍是 engineering／exploratory evidence，strict reference 未驗證。** 新 collector port 尚未執行三設備新 GPU smoke；既有資料數值重播不等於 live capture 相容性驗收。此次沒有新增 GPU 量測或研究資料重訓。

## 開始使用

```bash
python -m pip install -e .
./run.sh -m unittest discover -s tests -v
./run.sh scripts/audit_i3_catalog.py
```

本機 `run.sh` 已設定可用的 Python 環境。其他節點可設 `GPU_PRE2_PYTHON`；資料根目錄預設為 Repo 外的 `../research_data`。完整命令、重播、資料布局與新收集入口見 [LOCAL_DATA_LAYOUT](docs/LOCAL_DATA_LAYOUT.md)。

## 遷移驗證

- 74 項 CPU 測試通過；synthetic mixed CLI 另作工程 smoke，不能當研究訓練成果。
- I2：126 traces、6,990 個解析產物逐 byte 一致。
- I3：672 traces、36,992 個產物；36,902 個逐 byte 一致，90 個 Groups4 predictor 檔僅修正四 stream aliases。
- I4：三 seed × 三 repeat 預測逐 byte 一致；target/reference poison 不改預測；12 份評估產物亦逐 byte 一致。
- 舊階段 import、sys.path 注入與自動資料搜尋均被移除／拒絕。來源快照以 hash blobs 保存證據，不加入執行路徑。

詳見 [遷移報告](docs/MIGRATION_REPORT_ZH_TW.md)、[遷移狀態](docs/MIGRATION_STATUS.md)、[資料契約](docs/DATA_CONTRACT.md)、[I4 mixed training](docs/I4_MIXED_TRAINING.md)。

Git 保存程式、配置、manifest、coverage 和精簡結果。Raw traces、固定 weights／tensors、I4 checkpoints 與完整來源證據在 Git 外；逐檔索引由 `data/manifests/ASSET_INDEX.json` 指向。既有缺類、Memset geometry、reference warning、stack 差異和單一 S4 context group 均保留。
