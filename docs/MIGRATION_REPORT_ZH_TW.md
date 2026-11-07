# 本機整合與清理報告

2026-11-07。完成 E2、I2、I3、O3 必要方法及 I4 歷史基準的本機整合。所有執行程式以 GPU_PRE2 為唯一來源；資料使用明確註冊 ID 和相對路徑。沒有執行 GPU 工作、沒有重訓研究模型或修改現有系統服務。

## 保留資產

- 三設備 E2 profiles、curves、microbench samples 及必要來源證據。
- I2 三設備七 A：189 cells、7,560 measured。
- I3 三設備選定資料：3,024 cells、120,960 measured，保留 recovery、backend 與品質差異。
- I4 三 seed recognizer + 三 seed N0、normalizers、frozen policy、split、完整 canonical v2 輸入／預測／精簡評估。
- O3 方法文件與一個 264 KB 的 L4 regression fixture；沒有保留舊 O3 實驗樹。

初次搬移 43 個明確登錄項目（含一個 E2 歷史別名），全部逐檔核對。別名已改為註冊表指向 canonical E2，沒有留下指向舊目錄的 symlink。最後外部 hash index 包含 225,641 份檔案；搬移前 246,899 份資產引用均已重核，包含轉為 inert source blobs 的來源證據。

## 驗證

74 項 CPU 測試通過；另完成 synthetic mixed-training CLI 工程 smoke。I2 126 traces／6,990 artifacts 逐 byte 一致。I3 672 traces／36,992 artifacts，36,902 逐 byte 一致；其餘 90 份 Groups4 predictor 投影只修正四 stream aliases，其他欄位一致。新 projection 不覆寫原始資料。

I4 三 seed × 三 repeat 共 1,008 筆預測與歷史 canonical v2 逐 byte 一致；移除 reference 並在磁碟放置 poison target/reference 檔後，預測 SHA 不變。12 份 evaluator 輸出亦逐 byte 一致。特徵使用歷史精確候選片段 E1 27 維；未把新的 event mean pooling 與舊權重混用。

實際 Task Coverage 為 7,200 列，包含兩個未歸屬 Qwen copy 事件。完整語意參數不足，僅標 ATen＋shape match；不宣稱完整 exact coverage 或模型泛化成績。

## 清理

已刪除 13 個舊本機根目錄：四份 O1/O2 worktree、舊 gpu-prediction checkout、O3、O4、artifacts、inputs、received、archives 和兩個空的 RTX archives 入口。保留範圍內的 bytes 已先遷移且驗證；原 checkout 的 tracked 修改以 patch 保存。舊來源快照壓成 226 個初始唯一 hash blobs，後續補入原 shell 腳本，不加入 Python 搜尋路徑。

清理當下工作區配置空間由 134.15 GiB 降至 13.19 GiB，減少約 **120.97 GiB**。之後報告／驗證 log 會增加少量空間。遠端舊 Repo、分支與工作區保護目錄沒有刪除。

逐路徑刪除紀錄、來源→目的地 journal、搬前 hash 索引及完整驗證證據在本機 `../runs/migration_20261107`。Git 內保存精簡結果與索引，raw／weights 留在外部資料根目錄。

## 保留的限制

所有 strict reference 仍未驗證。缺少的 3090 isolated operand 證據與 4090 smoke、軟體 stack／UUID 相容性、GB10 時間戳／overlap、Groups4 all-stream GI completion、Memset geometry、缺類及單一 S4 context group 均未解除。新 collector live smoke、實際資料的新 mixed trainer converter、獨立 context 驗證和 I5 屬後續研究。
