# 交通資料品質檢查指南

## 概述

本文檔說明交通資料的品質標準、常見問題、檢查方法和解決方案。

## 資料品質標準

### 預期品質指標

| 指標 | 正常範圍 | 優良範圍 | 警告範圍 | 說明 |
|------|----------|----------|----------|------|
| **VD 匹配率** | 80-90% | >90% | <80% | VDList 與 VDLiveList 的 ID 匹配比例 |
| **有效資料比例** | 60-85% | >85% | <60% | 過濾錯誤碼後的有效數值比例 |
| **NaN 比例** | 15-40% | <15% | >40% | 包含錯誤碼過濾和缺失資料 |
| **時間覆蓋率** | >95% | >98% | <95% | 連續時間步的資料完整性 |
| **空間覆蓋率** | >80% | >90% | <80% | 地理區域內 VD 的資料完整性 |

### 特徵範圍標準

| 特徵 | 正常範圍 | 典型值 | 異常閾值 | 處理方式 |
|------|----------|--------|----------|----------|
| **avg_speed** | 0-120 km/h | 20-80 km/h | >200 或 <0 | 過濾異常值 |
| **total_volume** | 0-100 輛/分 | 0-20 輛/分 | >500 或 <0 | 檢查聚合邏輯 |
| **avg_occupancy** | 0-50% | 0-10% | >100 或 <0 | 過濾異常值 |
| **speed_std** | 0-50 km/h | 0-20 km/h | >100 | 檢查計算邏輯 |
| **lane_count** | 1-6 | 2-3 | >10 或 <1 | 檢查資料結構 |

## 錯誤碼識別

### 常見錯誤模式

| 錯誤碼/模式 | 出現原因 | 識別方法 | 處理策略 |
|-------------|----------|----------|----------|
| **-99** | 感測器故障 | 精確匹配 | 轉換為 NaN |
| **-1** | 初始化錯誤 | 精確匹配 | 轉換為 NaN |
| **255** | 數值溢出 | 精確匹配 | 轉換為 NaN |
| **999** | 系統錯誤 | 精確匹配 | 轉換為 NaN |
| **重複數值** | 系統卡死 | 統計檢測 | 檢查時間序列 |
| **突發異常** | 通訊錯誤 | 異常值檢測 | 平滑處理 |

### 錯誤碼檢測腳本

```python
def detect_error_codes(data, feature_name):
    """檢測並回報錯誤碼。"""
    import numpy as np
    
    # 常見錯誤碼
    error_codes = [-99, -1, 255, 999]
    error_stats = {}
    
    for code in error_codes:
        count = np.sum(data == code)
        if count > 0:
            error_stats[code] = {
                'count': count,
                'percentage': count / data.size * 100
            }
    
    # 檢查異常範圍
    if feature_name == 'speed':
        invalid_count = np.sum((data < -200) | (data > 300))
    elif feature_name == 'occupancy':
        invalid_count = np.sum((data < -200) | (data > 200))
    else:
        invalid_count = np.sum((data < -1000) | (data > 1000))
    
    if invalid_count > 0:
        error_stats['out_of_range'] = {
            'count': invalid_count,
            'percentage': invalid_count / data.size * 100
        }
    
    return error_stats
```

## 資料品質檢查工具

### 1. H5 檔案品質檢查

```python
import h5py
import numpy as np

def comprehensive_quality_check(h5_file_path):
    """全面的 H5 檔案品質檢查。"""
    
    with h5py.File(h5_file_path, 'r') as f:
        features = f['data/features']
        feature_names = [name.decode() for name in f['metadata/feature_names']]
        timestamps = [ts.decode() for ts in f['metadata/timestamps']]
        
        print(f"=== H5 檔案品質報告: {h5_file_path} ===")
        print(f"資料維度: {features.shape}")
        print(f"時間範圍: {timestamps[0]} 到 {timestamps[-1]}")
        print(f"特徵: {feature_names}")
        
        # 整體統計
        total_values = features.size
        nan_values = np.sum(np.isnan(features))
        valid_values = total_values - nan_values
        
        print(f"\n=== 整體統計 ===")
        print(f"總數值: {total_values:,}")
        print(f"有效值: {valid_values:,} ({valid_values/total_values:.1%})")
        print(f"NaN 值: {nan_values:,} ({nan_values/total_values:.1%})")
        
        # 各特徵統計
        print(f"\n=== 各特徵品質 ===")
        for i, fname in enumerate(feature_names):
            feature_data = features[:, :, i]
            valid_data = feature_data[~np.isnan(feature_data)]
            
            print(f"\n{fname}:")
            print(f"  有效值: {len(valid_data):,}/{feature_data.size:,} ({len(valid_data)/feature_data.size:.1%})")
            
            if len(valid_data) > 0:
                print(f"  範圍: {valid_data.min():.2f} - {valid_data.max():.2f}")
                print(f"  平均: {valid_data.mean():.2f}")
                print(f"  標準差: {valid_data.std():.2f}")
                
                # 異常值檢查
                if fname == 'avg_speed':
                    anomalies = np.sum((valid_data < 0) | (valid_data > 200))
                elif fname == 'avg_occupancy':
                    anomalies = np.sum((valid_data < 0) | (valid_data > 100))
                elif fname in ['total_volume', 'lane_count']:
                    anomalies = np.sum(valid_data < 0)
                else:
                    anomalies = 0
                
                if anomalies > 0:
                    print(f"  ⚠️ 異常值: {anomalies} ({anomalies/len(valid_data):.1%})")
                else:
                    print(f"  ✅ 無異常值")
        
        # VD 覆蓋率
        print(f"\n=== VD 覆蓋率 ===")
        locations_with_data = 0
        for vd_idx in range(features.shape[1]):
            vd_data = features[:, vd_idx, :]
            if not np.all(np.isnan(vd_data)):
                locations_with_data += 1
        
        coverage = locations_with_data / features.shape[1]
        print(f"有資料的 VD: {locations_with_data}/{features.shape[1]} ({coverage:.1%})")
        
        # 時間完整性
        print(f"\n=== 時間完整性 ===")
        time_completeness = []
        for t in range(features.shape[0]):
            timestep_data = features[t, :, :]
            valid_ratio = np.sum(~np.isnan(timestep_data)) / timestep_data.size
            time_completeness.append(valid_ratio)
        
        avg_completeness = np.mean(time_completeness)
        min_completeness = np.min(time_completeness)
        print(f"平均完整性: {avg_completeness:.1%}")
        print(f"最低完整性: {min_completeness:.1%}")
        
        # 品質等級評估
        print(f"\n=== 品質評估 ===")
        quality_score = 0
        
        if coverage >= 0.9:
            quality_score += 25
            print("✅ VD 覆蓋率: 優秀")
        elif coverage >= 0.8:
            quality_score += 15
            print("⚠️ VD 覆蓋率: 良好")
        else:
            quality_score += 5
            print("❌ VD 覆蓋率: 需改善")
        
        if valid_values/total_values >= 0.85:
            quality_score += 25
            print("✅ 有效資料比例: 優秀")
        elif valid_values/total_values >= 0.6:
            quality_score += 15
            print("⚠️ 有效資料比例: 良好")
        else:
            quality_score += 5
            print("❌ 有效資料比例: 需改善")
        
        if avg_completeness >= 0.95:
            quality_score += 25
            print("✅ 時間完整性: 優秀")
        elif avg_completeness >= 0.8:
            quality_score += 15
            print("⚠️ 時間完整性: 良好")
        else:
            quality_score += 5
            print("❌ 時間完整性: 需改善")
        
        # 異常值檢查
        total_anomalies = 0
        for i, fname in enumerate(feature_names):
            feature_data = features[:, :, i]
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0:
                if fname == 'avg_speed':
                    total_anomalies += np.sum((valid_data < 0) | (valid_data > 200))
                elif fname == 'avg_occupancy':
                    total_anomalies += np.sum((valid_data < 0) | (valid_data > 100))
        
        if total_anomalies == 0:
            quality_score += 25
            print("✅ 無異常值")
        elif total_anomalies < valid_values * 0.01:
            quality_score += 15
            print("⚠️ 少量異常值")
        else:
            quality_score += 5
            print("❌ 異常值過多")
        
        print(f"\n=== 總體品質評分: {quality_score}/100 ===")
        if quality_score >= 80:
            print("🎉 資料品質優秀，可直接用於訓練")
        elif quality_score >= 60:
            print("⚠️ 資料品質良好，建議進一步清理")
        else:
            print("❌ 資料品質不佳，需要重新處理")

# 使用範例
comprehensive_quality_check('blob/dataset/pre-processed/h5/traffic_features.h5')
```

### 2. 原始 JSON 品質檢查

```python
def check_json_quality(json_dir_path):
    """檢查原始 JSON 資料品質。"""
    from social_xlstm.dataset.utils.json_utils import VDLiveList
    import os
    
    time_dirs = sorted([d for d in os.listdir(json_dir_path) 
                       if os.path.isdir(os.path.join(json_dir_path, d))])
    
    print(f"=== JSON 資料品質檢查: {json_dir_path} ===")
    print(f"時間目錄數量: {len(time_dirs)}")
    
    # 抽樣檢查
    sample_dirs = time_dirs[::max(1, len(time_dirs)//10)]  # 取10%樣本
    
    error_code_stats = {}
    vd_count_stats = []
    data_availability_stats = []
    
    for time_dir in sample_dirs:
        vd_live_file = os.path.join(json_dir_path, time_dir, 'VDLiveList.json')
        
        try:
            vd_live_list = VDLiveList.load_from_json(vd_live_file)
            vd_count_stats.append(len(vd_live_list.LiveTrafficData))
            
            vds_with_data = 0
            speed_values = []
            occupancy_values = []
            volume_values = []
            
            for vd in vd_live_list.LiveTrafficData:
                has_data = False
                if vd.LinkFlows:
                    for link in vd.LinkFlows:
                        if link.Lanes:
                            has_data = True
                            for lane in link.Lanes:
                                speed_values.append(lane.Speed)
                                occupancy_values.append(lane.Occupancy)
                                if lane.Vehicles:
                                    for vehicle in lane.Vehicles:
                                        volume_values.append(vehicle.Volume)
                
                if has_data:
                    vds_with_data += 1
            
            data_availability_stats.append(vds_with_data / len(vd_live_list.LiveTrafficData))
            
            # 錯誤碼統計
            for values, name in [(speed_values, 'speed'), 
                               (occupancy_values, 'occupancy'), 
                               (volume_values, 'volume')]:
                error_stats = detect_error_codes(np.array(values), name)
                for code, stats in error_stats.items():
                    key = f"{name}_{code}"
                    if key not in error_code_stats:
                        error_code_stats[key] = []
                    error_code_stats[key].append(stats['percentage'])
                    
        except Exception as e:
            print(f"⚠️ 處理 {time_dir} 時發生錯誤: {e}")
    
    # 報告結果
    print(f"\n=== VD 數量統計 ===")
    print(f"平均 VD 數量: {np.mean(vd_count_stats):.0f}")
    print(f"VD 數量範圍: {np.min(vd_count_stats)} - {np.max(vd_count_stats)}")
    
    print(f"\n=== 資料可用性 ===")
    print(f"平均有資料的 VD 比例: {np.mean(data_availability_stats):.1%}")
    print(f"最低/最高比例: {np.min(data_availability_stats):.1%} / {np.max(data_availability_stats):.1%}")
    
    print(f"\n=== 錯誤碼統計 ===")
    for key, percentages in error_code_stats.items():
        avg_percentage = np.mean(percentages)
        if avg_percentage > 0.1:  # 只顯示 >0.1% 的錯誤
            print(f"{key}: 平均 {avg_percentage:.1%} (範圍: {np.min(percentages):.1%}-{np.max(percentages):.1%})")
```

## 常見品質問題與解決方案

### 問題 1: VD 匹配率過低 (<80%)

**可能原因**：
- VDList 和 VDLiveList 來源不同步
- 時間範圍選擇問題
- 部分區域 VD 大量故障

**解決方案**：
```python
# 檢查不匹配的 VD
missing_vds = set(vd_list_ids) - set(vd_live_ids)
extra_vds = set(vd_live_ids) - set(vd_list_ids)

# 分析地理分布
for vd_id in missing_vds:
    vd_info = find_vd_info(vd_id)
    print(f"缺失 VD: {vd_id}, 位置: {vd_info.RoadName}")
```

### 問題 2: 異常值過多

**可能原因**：
- 錯誤碼過濾不完整
- 新的錯誤碼類型
- 感測器校準問題

**解決方案**：
```python
# 擴展錯誤碼檢測
def enhanced_error_detection(data):
    # 檢查統計異常
    q99 = np.percentile(data, 99)
    q01 = np.percentile(data, 1)
    
    # 可能的新錯誤碼
    potential_errors = []
    for value in np.unique(data):
        count = np.sum(data == value)
        if count > len(data) * 0.05:  # 出現頻率 >5%
            potential_errors.append(value)
    
    return potential_errors
```

### 問題 3: 時間完整性不足

**可能原因**：
- 網路傳輸中斷
- 系統維護時段
- 資料壓縮檔遺失

**解決方案**：
```python
# 檢查時間間隔
def check_time_gaps(timestamps):
    import pandas as pd
    
    timestamps_dt = pd.to_datetime(timestamps)
    time_diffs = timestamps_dt.diff()[1:]  # 跳過第一個 NaT
    
    expected_interval = pd.Timedelta(minutes=5)  # 假設5分鐘間隔
    large_gaps = time_diffs[time_diffs > expected_interval * 2]
    
    print(f"發現 {len(large_gaps)} 個時間間隔異常")
    for gap in large_gaps:
        print(f"異常間隔: {gap}")
```

## 自動化品質監控

### 建立品質監控腳本
```python
#!/usr/bin/env python3
"""
交通資料品質監控腳本
定期檢查資料品質並生成報告
"""

def automated_quality_monitor():
    import os
    import datetime
    
    # 配置
    h5_file = 'blob/dataset/pre-processed/h5/traffic_features.h5'
    json_dir = 'blob/dataset/pre-processed/unzip_to_json'
    report_dir = 'blob/reports/quality'
    
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成報告
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"{report_dir}/quality_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        # 重定向輸出到檔案
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print(f"品質檢查報告 - {datetime.datetime.now()}")
            print("=" * 50)
            
            if os.path.exists(h5_file):
                comprehensive_quality_check(h5_file)
            
            if os.path.exists(json_dir):
                check_json_quality(json_dir)
                
        finally:
            sys.stdout = original_stdout
    
    print(f"品質報告已生成: {report_file}")

if __name__ == "__main__":
    automated_quality_monitor()
```

## 品質改善建議

### 短期改善 (1-2週)
1. **完善錯誤碼過濾**：擴展已知錯誤碼清單
2. **異常值檢測**：實施統計型異常值檢測
3. **即時監控**：建立每日品質檢查報告

### 中期改善 (1-2個月)
1. **智能填補**：使用時間序列插值填補缺失值
2. **空間校正**：利用相鄰 VD 進行資料校正
3. **感測器健康監控**：追蹤個別 VD 的穩定性

### 長期改善 (3-6個月)
1. **機器學習清理**：訓練異常檢測模型
2. **多源融合**：整合其他交通資料源
3. **預測性維護**：預測感測器故障

## 相關工具與腳本

- [品質檢查腳本](../scripts/utils/quality_check.py)
- [錯誤碼統計工具](../scripts/utils/error_code_analysis.py)
- [自動化監控系統](../scripts/monitoring/quality_monitor.py)