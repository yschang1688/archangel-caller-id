# Python 環境快速切換指南

本指南說明如何在 Python 3.11（condaml）與 Python 3.13（base）之間快速切換，以及如何在 IDE 中確認當前使用的版本。

> **更新：** 環境已從 `fraudml` 重新命名為 `condaml`，名稱更通用，適合所有機器學習專案。

---

## 📋 環境概覽

| 環境名稱 | Python 版本 | 路徑 | 用途 |
|---------|------------|------|------|
| **base** | Python 3.13 | `/opt/anaconda3/bin/python` | Anaconda 預設環境 |
| **condaml** | Python 3.11 | `/opt/anaconda3/envs/condaml/bin/python` | 機器學習專案（已安裝所有 ML 套件） |
| ~~**fraudml**~~ | ~~Python 3.11~~ | ~~`/opt/anaconda3/envs/fraudml/bin/python`~~ | ~~已重新命名為 condaml~~ |

---

## 🖥️ 方法一：在終端機中切換

### 切換到 Python 3.11（condaml 環境）

```bash
conda activate condaml
```

**驗證版本：**
```bash
python --version
# 應該顯示：Python 3.11.x
```

**確認路徑：**
```bash
which python
# 應該顯示：/opt/anaconda3/envs/condaml/bin/python
```

### 切換到 Python 3.13（base 環境）

```bash
conda activate base
# 或直接
conda deactivate  # 如果當前在 fraudml 環境中
```

**驗證版本：**
```bash
python --version
# 應該顯示：Python 3.13.x
```

**確認路徑：**
```bash
which python
# 應該顯示：/opt/anaconda3/bin/python
```

### 快速檢查當前環境

```bash
# 方法 1：查看 conda 環境列表（當前環境會有 * 標記）
conda env list

# 方法 2：查看環境變數
echo $CONDA_DEFAULT_ENV

# 方法 3：查看 Python 路徑
which python
```

---

## 💻 方法二：在 VS Code / Cursor IDE 中切換

### 步驟 1：打開 Python 解譯器選擇器

**方法 A - 使用命令面板：**
1. 按 `Cmd+Shift+P`（Mac）或 `Ctrl+Shift+P`（Windows/Linux）
2. 輸入 `Python: Select Interpreter`
3. 選擇對應的 Python 版本

**方法 B - 使用狀態列：**
1. 點擊 VS Code/Cursor 右下角的 Python 版本顯示（例如：`Python 3.11.13`）
2. 選擇 `Python: Select Interpreter`

### 步驟 2：選擇對應的環境

在彈出的列表中選擇：

- **Python 3.11（condaml）**：
  - 選擇 `Python 3.11.13 ('condaml': conda)`
  - 或手動輸入：`/opt/anaconda3/envs/condaml/bin/python`

- **Python 3.13（base）**：
  - 選擇 `Python 3.13.x ('base': conda)`
  - 或手動輸入：`/opt/anaconda3/bin/python`

### 步驟 3：確認當前版本

**在 IDE 狀態列查看：**
- 右下角會顯示當前選擇的 Python 版本，例如：
  - `Python 3.11.13 ('condaml': conda)` ← Python 3.11
  - `Python 3.13.x ('base': conda)` ← Python 3.13

**在終端機中驗證（IDE 內建終端機）：**
```bash
python --version
which python
```

---

## 🔍 如何確認當前使用的 Python 版本

### 在終端機中

```bash
# 方法 1：查看版本
python --version

# 方法 2：查看詳細資訊
python -c "import sys; print(sys.version)"

# 方法 3：查看路徑（最準確）
which python
# 或
python -c "import sys; print(sys.executable)"
```

### 在 IDE 中

**方法 1：狀態列**
- 查看右下角的 Python 版本顯示
- 格式：`Python 3.11.13 ('fraudml': conda)`

**方法 2：命令面板**
1. 按 `Cmd+Shift+P`（Mac）或 `Ctrl+Shift+P`（Windows/Linux）
2. 輸入 `Python: Show Interpreter`
3. 會顯示當前選擇的解譯器路徑

**方法 3：終端機面板**
- 在 IDE 內建的終端機中執行：
```bash
python --version
which python
```

**方法 4：查看設定檔**
- 專案根目錄下的 `.vscode/settings.json` 會記錄當前選擇的解譯器：
```json
{
    "python.defaultInterpreterPath": "/opt/anaconda3/envs/fraudml/bin/python"
}
```

---

## ⚡ 快速切換技巧

### 技巧 1：使用快捷鍵

在 VS Code/Cursor 中：
- `Cmd+Shift+P`（Mac）或 `Ctrl+Shift+P`（Windows/Linux）
- 輸入 `interpreter` 快速找到 Python 解譯器選項

### 技巧 2：為專案設定預設解譯器

在專案根目錄建立 `.vscode/settings.json`：

**設定為 Python 3.11（fraudml）：**
```json
{
    "python.defaultInterpreterPath": "/opt/anaconda3/envs/fraudml/bin/python"
}
```

**設定為 Python 3.13（base）：**
```json
{
    "python.defaultInterpreterPath": "/opt/anaconda3/bin/python"
}
```

### 技巧 3：使用終端機快捷指令

在 `~/.zshrc` 或 `~/.bashrc` 中加入別名：

```bash
# 快速切換到 condaml 環境
alias py311='conda activate condaml'

# 快速切換到 base 環境
alias py313='conda activate base'

# 查看當前 Python 版本
alias pyversion='python --version && which python'
```

然後執行：
```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

之後就可以直接使用：
```bash
py311    # 切換到 Python 3.11
py313    # 切換到 Python 3.13
pyversion # 查看當前版本
```

---

## 🎯 推薦使用場景

### 使用 Python 3.11（condaml）當：
- ✅ 開發機器學習專案
- ✅ 需要使用 xgboost、shap、imbalanced-learn 等 ML 套件
- ✅ 執行 `clean_and_prepare_data.py`、`train_xgboost_and_evaluate.py` 等腳本
- ✅ 進行防詐 DRE 面試 POC 開發

### 使用 Python 3.13（base）當：
- ✅ 一般 Python 開發
- ✅ 測試新版本 Python 特性
- ✅ 不需要特定 ML 套件的專案

---

## ⚠️ 常見問題

### Q1: IDE 顯示的版本與終端機不一致？

**原因：** IDE 使用專案設定的解譯器，終端機使用當前激活的 conda 環境。

**解決方法：**
- 確保 IDE 選擇的解譯器與終端機激活的環境一致
- 或在 IDE 終端機中執行 `conda activate <環境名稱>`

### Q2: 如何讓 IDE 自動使用當前終端機的環境？

在 `.vscode/settings.json` 中設定：
```json
{
    "python.terminal.activateEnvironment": true
}
```

### Q3: 切換環境後套件找不到？

**檢查清單：**
1. 確認當前 Python 版本：`python --version`
2. 確認 Python 路徑：`which python`
3. 確認套件是否安裝在當前環境：`pip list` 或 `conda list`
4. 確認 IDE 選擇的解譯器路徑正確

---

## 📝 快速參考表

| 操作 | Python 3.11（fraudml） | Python 3.13（base） |
|------|----------------------|-------------------|
| **終端機切換** | `conda activate condaml` | `conda activate base` |
| **IDE 解譯器路徑** | `/opt/anaconda3/envs/condaml/bin/python` | `/opt/anaconda3/bin/python` |
| **查看版本** | `python --version` | `python --version` |
| **查看路徑** | `which python` | `which python` |
| **IDE 狀態列顯示** | `Python 3.11.13 ('condaml': conda)` | `Python 3.13.x ('base': conda)` |

---

**最後更新：** 2025-01-14
