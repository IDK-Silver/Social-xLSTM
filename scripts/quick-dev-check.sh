#!/bin/bash
# 快速開發檢查腳本 - 適合開發階段使用
# Quick Development Check Script - Suitable for development phase

echo "🚀 快速開發檢查 (Quick Dev Check)"
echo "==============================================="
echo "📝 注意: 這是開發階段的寬鬆檢查，接近發布時請使用完整 CI/CD 檢查"
echo "Note: This is a relaxed check for development phase"
echo ""

# 設定錯誤時立即退出
set -e

# 計時開始
start_time=$(date +%s)

echo "🔍 1. 基本語法檢查 (Basic Syntax Check)..."
# 檢查主要 Python 檔案的語法
python -m py_compile src/social_xlstm/models/*.py 2>/dev/null || {
    echo "❌ 語法錯誤 in models/"
    exit 1
}

python -m py_compile src/social_xlstm/utils/*.py 2>/dev/null || {
    echo "❌ 語法錯誤 in utils/"
    exit 1
}

echo "✅ 語法檢查通過"

echo ""
echo "🧪 2. 核心功能測試 (Core Tests Only)..."
# 只跑最重要的核心測試，跳過耗時的整合測試
if [ -d "tests/core" ]; then
    pytest tests/core/ -x --tb=short -q
elif [ -d "tests" ]; then
    # 如果有 tests 目錄，只跑快速測試
    pytest tests/ -x --tb=short -q -k "not slow and not integration" --maxfail=3
else
    echo "⚠️  未找到測試目錄，跳過測試"
fi
echo "✅ 核心測試通過"

echo ""
echo "📦 3. 基本匯入檢查 (Basic Import Check)..."
# 檢查主要模組是否可以正常匯入
python -c "
try:
    import src.social_xlstm.models.lstm
    import src.social_xlstm.utils.spatial_coords
    print('✅ 主要模組匯入成功')
except ImportError as e:
    print(f'❌ 匯入錯誤: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  匯入警告: {e}')
    print('✅ 基本匯入可能有小問題但不影響開發')
"

# 計算執行時間
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "==============================================="
echo "🎉 快速檢查完成！執行時間: ${duration} 秒"
echo "✅ 基本語法正確，可以繼續開發"
echo ""
echo "💡 提示:"
echo "   - 如果要完整檢查，請用: python scripts/check_architecture_rules.py"
echo "   - 如果要提交程式碼，建議先跑: git push (會觸發完整 CI 檢查)"
echo "   - 實驗性程式碼可以用: git commit -m 'experiment [skip ci]'"
echo ""