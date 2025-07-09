#!/usr/bin/env python3
"""
Claude Code 初始化腳本

自動讀取並展示專案關鍵信息，幫助新的 Claude Code 會話快速了解專案狀態。

使用方法:
    python scripts/utils/claude_init.py
    python scripts/utils/claude_init.py --full  # 完整模式
    python scripts/utils/claude_init.py --quick # 快速模式
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def print_header(title: str, level: int = 1):
    """打印格式化的標題"""
    chars = ['=', '-', '·']
    char = chars[min(level-1, len(chars)-1)]
    print(f"\n{char * 60}")
    print(f"{title}")
    print(f"{char * 60}")

def read_file_safely(file_path: str) -> str:
    """安全讀取文件內容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"❌ 文件不存在: {file_path}"
    except Exception as e:
        return f"❌ 讀取錯誤: {e}"

def show_project_status():
    """顯示專案狀態"""
    print_header("📊 專案狀態概覽")
    
    status_file = "docs/reports/project_status.md"
    content = read_file_safely(status_file)
    
    # 提取執行摘要
    lines = content.split('\n')
    in_summary = False
    summary_lines = []
    
    for line in lines:
        if line.strip().startswith('## 📊 執行摘要'):
            in_summary = True
            continue
        elif in_summary and line.strip().startswith('## '):
            break
        elif in_summary:
            summary_lines.append(line)
    
    if summary_lines:
        print('\n'.join(summary_lines))
    else:
        print("❌ 無法讀取專案狀態")

def show_adr_status():
    """顯示 ADR 狀態"""
    print_header("🏛️ ADR 架構決策狀態")
    
    adr_readme = "docs/adr/README.md"
    content = read_file_safely(adr_readme)
    
    # 提取 ADR 清單
    lines = content.split('\n')
    in_list = False
    
    for line in lines:
        if '### 核心架構' in line:
            in_list = True
            print("### 核心架構 (0001-0099)")
        elif '### 技術選擇' in line:
            print("\n### 技術選擇 (0100-0199)")
        elif '### 座標系統' in line:
            print("\n### 座標系統 (0200-0299)")
        elif '### 開發流程' in line:
            print("\n### 開發流程 (0300-0399)")
        elif '### 實施細節' in line:
            print("\n### 實施細節 (0400-0499)")
        elif in_list and line.strip().startswith('- [ADR-'):
            print(line)
        elif in_list and line.strip().startswith('## '):
            break

def show_current_priorities():
    """顯示當前開發重點"""
    print_header("🎯 當前開發重點")
    
    # 讀取 ADR-0300 的優先級
    adr_300 = "docs/adr/0300-next-development-priorities.md"
    content = read_file_safely(adr_300)
    
    print("基於 ADR-0300 決策:")
    print("✅ 架構清理階段已完成")
    print("📋 當前重點: 核心功能開發")
    print()
    print("下一步開發目標:")
    print("1. 🎯 Social Pooling 算法實現 (ADR-0100)")
    print("2. 🧠 xLSTM 整合 (ADR-0101)")
    print("3. 🚀 Social-xLSTM 完整模型")
    print("4. 🧪 實驗驗證與評估")

def show_tech_decisions():
    """顯示核心技術決策"""
    print_header("🔧 核心技術決策")
    
    decisions = [
        ("ADR-0100", "Social Pooling vs Graph Networks", "選擇 Social Pooling 方法"),
        ("ADR-0101", "xLSTM vs Traditional LSTM", "選擇 xLSTM 混合架構"),
        ("ADR-0200", "座標系統選擇", "使用墨卡托投影系統"),
    ]
    
    for adr_id, title, decision in decisions:
        print(f"• {adr_id}: {title}")
        print(f"  → 決策: {decision}")
        print()

def show_code_structure():
    """顯示程式碼結構重點"""
    print_header("💻 程式碼結構重點")
    
    key_files = [
        ("src/social_xlstm/models/lstm.py", "統一的 LSTM 實現"),
        ("src/social_xlstm/utils/spatial_coords.py", "座標系統實現"),
        ("src/social_xlstm/training/trainer.py", "統一訓練系統"),
        ("src/social_xlstm/evaluation/evaluator.py", "模型評估框架"),
    ]
    
    print("📁 關鍵檔案:")
    for file_path, description in key_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - {description}")
        else:
            print(f"❌ {file_path} - {description} (文件不存在)")
    
    print("\n📋 訓練腳本:")
    training_scripts = [
        "scripts/train/train_single_vd.py",
        "scripts/train/train_multi_vd.py", 
        "scripts/train/test_training_scripts.py",
        "scripts/train/common.py"
    ]
    
    for script in training_scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} (文件不存在)")

def show_implementation_status():
    """顯示實施狀態"""
    print_header("📈 實施狀態")
    
    print("✅ 已完成:")
    print("  • LSTM 實現統一 (5→1 個實現)")
    print("  • 訓練腳本重構 (減少 48% 重複)")
    print("  • ADR 系統建立 (7 個核心決策)")
    print("  • 座標系統實現")
    print("  • 評估框架建立")
    
    print("\n🚧 進行中:")
    print("  • 專案結構整理")
    print("  • 文檔系統完善")
    
    print("\n📋 待開發:")
    print("  • Social Pooling 算法")
    print("  • xLSTM 整合")
    print("  • Social-xLSTM 模型")
    print("  • 實驗驗證")

def show_quick_commands():
    """顯示快速命令"""
    print_header("⚡ 快速命令")
    
    print("🚀 開發環境:")
    print("  conda activate social_xlstm")
    print("  pip install -e .")
    print()
    print("🧪 測試訓練:")
    print("  python scripts/train/test_training_scripts.py --quick")
    print()
    print("📊 查看數據:")
    print("  snakemake --cores 4")
    print()
    print("📖 查看文檔:")
    print("  cat docs/adr/README.md")
    print("  cat docs/todo.md")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Claude Code 初始化助手")
    parser.add_argument("--full", action="store_true", help="完整模式")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    
    args = parser.parse_args()
    
    print_header("🤖 Claude Code 初始化", 1)
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        show_current_priorities()
        show_tech_decisions()
        show_quick_commands()
    else:
        show_project_status()
        show_adr_status()
        show_current_priorities()
        show_tech_decisions()
        show_implementation_status()
        show_code_structure()
        show_quick_commands()
    
    print_header("🎯 建議的下一步動作", 2)
    print("1. 檢查 conda 環境: conda activate social_xlstm")
    print("2. 查看待辦事項: cat docs/todo.md")
    print("3. 開始 Social Pooling 實現")
    print("4. 參考 ADR-0100 和 ADR-0101 技術決策")
    
    print("\n" + "="*60)
    print("✅ 初始化完成 - 可以開始開發工作")
    print("="*60)

if __name__ == "__main__":
    main()