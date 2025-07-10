#!/usr/bin/env python3
"""
交通資料品質檢查工具

使用方式:
    python scripts/utils/quality_check.py --h5_file blob/dataset/pre-processed/h5/traffic_features.h5
    python scripts/utils/quality_check.py --json_dir blob/dataset/pre-processed/unzip_to_json --sample_size 10
    python scripts/utils/quality_check.py --all  # 檢查所有
"""

import argparse
import h5py
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from social_xlstm.dataset.utils.json_utils import VDLiveList


def detect_error_codes(data, feature_name):
    """檢測並回報錯誤碼。"""
    if len(data) == 0:
        return {}
        
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


def comprehensive_h5_quality_check(h5_file_path):
    """全面的 H5 檔案品質檢查。"""
    
    if not os.path.exists(h5_file_path):
        print(f"❌ H5 檔案不存在: {h5_file_path}")
        return
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            features = f['data/features']
            feature_names = [name.decode() for name in f['metadata/feature_names']]
            timestamps = [ts.decode() for ts in f['metadata/timestamps']]
            
            print(f"=== H5 檔案品質報告: {h5_file_path} ===")
            print(f"資料維度: {features.shape} (時間步, VD數, 特徵數)")
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
            anomaly_total = 0
            
            for i, fname in enumerate(feature_names):
                feature_data = features[:, :, i]
                valid_data = feature_data[~np.isnan(feature_data)]
                
                print(f"\n{fname}:")
                print(f"  有效值: {len(valid_data):,}/{feature_data.size:,} ({len(valid_data)/feature_data.size:.1%})")
                
                if len(valid_data) > 0:
                    print(f"  範圍: {valid_data.min():.2f} - {valid_data.max():.2f}")
                    print(f"  平均: {valid_data.mean():.2f}")
                    print(f"  標準差: {valid_data.std():.2f}")
                    print(f"  中位數: {np.median(valid_data):.2f}")
                    
                    # 異常值檢查
                    anomalies = 0
                    if fname == 'avg_speed':
                        anomalies = np.sum((valid_data < 0) | (valid_data > 200))
                    elif fname == 'avg_occupancy':
                        anomalies = np.sum((valid_data < 0) | (valid_data > 100))
                    elif fname in ['total_volume', 'lane_count']:
                        anomalies = np.sum(valid_data < 0)
                    elif fname == 'speed_std':
                        anomalies = np.sum(valid_data < 0)
                    
                    anomaly_total += anomalies
                    
                    if anomalies > 0:
                        print(f"  ⚠️ 異常值: {anomalies} ({anomalies/len(valid_data):.1%})")
                    else:
                        print(f"  ✅ 無異常值")
                        
                    # 分布統計
                    q25, q75 = np.percentile(valid_data, [25, 75])
                    print(f"  四分位範圍: {q25:.2f} - {q75:.2f}")
                        
                else:
                    print(f"  ❌ 無有效資料")
            
            # VD 覆蓋率分析
            print(f"\n=== VD 覆蓋率分析 ===")
            locations_with_data = 0
            vd_data_quality = []
            
            for vd_idx in range(features.shape[1]):
                vd_data = features[:, vd_idx, :]
                valid_count = np.sum(~np.isnan(vd_data))
                total_count = vd_data.size
                
                if valid_count > 0:
                    locations_with_data += 1
                    quality_ratio = valid_count / total_count
                    vd_data_quality.append(quality_ratio)
            
            coverage = locations_with_data / features.shape[1]
            print(f"有資料的 VD: {locations_with_data}/{features.shape[1]} ({coverage:.1%})")
            
            if vd_data_quality:
                avg_vd_quality = np.mean(vd_data_quality)
                print(f"VD 平均資料完整性: {avg_vd_quality:.1%}")
                print(f"VD 資料完整性範圍: {np.min(vd_data_quality):.1%} - {np.max(vd_data_quality):.1%}")
            
            # 時間完整性分析
            print(f"\n=== 時間完整性分析 ===")
            time_completeness = []
            time_quality_by_feature = {fname: [] for fname in feature_names}
            
            for t in range(features.shape[0]):
                timestep_data = features[t, :, :]
                valid_ratio = np.sum(~np.isnan(timestep_data)) / timestep_data.size
                time_completeness.append(valid_ratio)
                
                # 各特徵的時間品質
                for i, fname in enumerate(feature_names):
                    feature_slice = features[t, :, i]
                    feature_valid_ratio = np.sum(~np.isnan(feature_slice)) / feature_slice.size
                    time_quality_by_feature[fname].append(feature_valid_ratio)
            
            avg_completeness = np.mean(time_completeness)
            min_completeness = np.min(time_completeness)
            max_completeness = np.max(time_completeness)
            
            print(f"平均時間完整性: {avg_completeness:.1%}")
            print(f"時間完整性範圍: {min_completeness:.1%} - {max_completeness:.1%}")
            
            # 各特徵時間穩定性
            print(f"\n=== 特徵時間穩定性 ===")
            for fname in feature_names:
                quality_values = time_quality_by_feature[fname]
                std_quality = np.std(quality_values)
                print(f"{fname}: 平均 {np.mean(quality_values):.1%}, 標準差 {std_quality:.3f}")
            
            # 品質等級評估
            print(f"\n=== 品質評估 ===")
            quality_score = 0
            max_score = 100
            
            # VD 覆蓋率評分 (25分)
            if coverage >= 0.9:
                quality_score += 25
                print("✅ VD 覆蓋率: 優秀 (≥90%)")
            elif coverage >= 0.8:
                quality_score += 18
                print("⚠️ VD 覆蓋率: 良好 (80-90%)")
            elif coverage >= 0.7:
                quality_score += 12
                print("⚠️ VD 覆蓋率: 普通 (70-80%)")
            else:
                quality_score += 5
                print("❌ VD 覆蓋率: 需改善 (<70%)")
            
            # 有效資料比例評分 (25分)
            valid_ratio = valid_values/total_values
            if valid_ratio >= 0.85:
                quality_score += 25
                print("✅ 有效資料比例: 優秀 (≥85%)")
            elif valid_ratio >= 0.7:
                quality_score += 18
                print("⚠️ 有效資料比例: 良好 (70-85%)")
            elif valid_ratio >= 0.6:
                quality_score += 12
                print("⚠️ 有效資料比例: 普通 (60-70%)")
            else:
                quality_score += 5
                print("❌ 有效資料比例: 需改善 (<60%)")
            
            # 時間完整性評分 (25分)
            if avg_completeness >= 0.95:
                quality_score += 25
                print("✅ 時間完整性: 優秀 (≥95%)")
            elif avg_completeness >= 0.85:
                quality_score += 18
                print("⚠️ 時間完整性: 良好 (85-95%)")
            elif avg_completeness >= 0.75:
                quality_score += 12
                print("⚠️ 時間完整性: 普通 (75-85%)")
            else:
                quality_score += 5
                print("❌ 時間完整性: 需改善 (<75%)")
            
            # 異常值評分 (25分)
            anomaly_ratio = anomaly_total / valid_values if valid_values > 0 else 1
            if anomaly_ratio == 0:
                quality_score += 25
                print("✅ 無異常值: 優秀")
            elif anomaly_ratio < 0.01:
                quality_score += 18
                print("⚠️ 異常值: 良好 (<1%)")
            elif anomaly_ratio < 0.05:
                quality_score += 12
                print("⚠️ 異常值: 普通 (1-5%)")
            else:
                quality_score += 5
                print("❌ 異常值: 需改善 (>5%)")
            
            print(f"\n=== 總體品質評分: {quality_score}/{max_score} ===")
            if quality_score >= 85:
                print("🎉 資料品質優秀，可直接用於訓練")
            elif quality_score >= 70:
                print("⚠️ 資料品質良好，建議進一步清理")
            elif quality_score >= 50:
                print("⚠️ 資料品質普通，需要清理和預處理")
            else:
                print("❌ 資料品質不佳，需要重新處理")
                
            # 建議
            print(f"\n=== 改善建議 ===")
            if coverage < 0.8:
                print("• 檢查 VD 匹配邏輯，確認 VDList 和 VDLiveList 同步")
            if valid_ratio < 0.7:
                print("• 檢查錯誤碼過濾邏輯，可能有新的錯誤模式")
            if avg_completeness < 0.8:
                print("• 檢查時間序列完整性，考慮時間間隔插值")
            if anomaly_ratio > 0.01:
                print("• 加強異常值檢測，檢查特徵提取邏輯")
                
    except Exception as e:
        print(f"❌ 檢查 H5 檔案時發生錯誤: {e}")


def check_json_quality(json_dir_path, sample_size=None):
    """檢查原始 JSON 資料品質。"""
    
    if not os.path.exists(json_dir_path):
        print(f"❌ JSON 目錄不存在: {json_dir_path}")
        return
    
    time_dirs = sorted([d for d in os.listdir(json_dir_path) 
                       if os.path.isdir(os.path.join(json_dir_path, d))])
    
    if not time_dirs:
        print(f"❌ JSON 目錄中沒有時間目錄: {json_dir_path}")
        return
    
    print(f"=== JSON 資料品質檢查: {json_dir_path} ===")
    print(f"時間目錄數量: {len(time_dirs)}")
    
    # 決定樣本大小
    if sample_size is None:
        sample_size = min(10, len(time_dirs))
    else:
        sample_size = min(sample_size, len(time_dirs))
    
    # 抽樣檢查
    if len(time_dirs) > sample_size:
        step = len(time_dirs) // sample_size
        sample_dirs = time_dirs[::step][:sample_size]
        print(f"抽樣檢查 {len(sample_dirs)} 個時間目錄 (每 {step} 個取一個)")
    else:
        sample_dirs = time_dirs
        print(f"檢查所有 {len(sample_dirs)} 個時間目錄")
    
    error_code_stats = {}
    vd_count_stats = []
    data_availability_stats = []
    processing_errors = []
    
    for i, time_dir in enumerate(sample_dirs):
        print(f"處理 {i+1}/{len(sample_dirs)}: {time_dir}", end=" ... ")
        
        vd_live_file = os.path.join(json_dir_path, time_dir, 'VDLiveList.json')
        
        if not os.path.exists(vd_live_file):
            print("❌ VDLiveList.json 不存在")
            processing_errors.append(f"{time_dir}: VDLiveList.json 不存在")
            continue
        
        try:
            vd_live_list = VDLiveList.load_from_json(vd_live_file)
            vd_count = len(vd_live_list.LiveTrafficData)
            vd_count_stats.append(vd_count)
            
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
                                if lane.Speed is not None:
                                    speed_values.append(lane.Speed)
                                if lane.Occupancy is not None:
                                    occupancy_values.append(lane.Occupancy)
                                if lane.Vehicles:
                                    for vehicle in lane.Vehicles:
                                        if vehicle.Volume is not None:
                                            volume_values.append(vehicle.Volume)
                
                if has_data:
                    vds_with_data += 1
            
            availability = vds_with_data / vd_count if vd_count > 0 else 0
            data_availability_stats.append(availability)
            
            # 錯誤碼統計
            for values, name in [(speed_values, 'speed'), 
                               (occupancy_values, 'occupancy'), 
                               (volume_values, 'volume')]:
                if values:
                    error_stats = detect_error_codes(np.array(values), name)
                    for code, stats in error_stats.items():
                        key = f"{name}_{code}"
                        if key not in error_code_stats:
                            error_code_stats[key] = []
                        error_code_stats[key].append(stats['percentage'])
                        
            print("✅")
                    
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            processing_errors.append(f"{time_dir}: {e}")
    
    # 報告結果
    if vd_count_stats:
        print(f"\n=== VD 數量統計 ===")
        print(f"平均 VD 數量: {np.mean(vd_count_stats):.0f}")
        print(f"VD 數量範圍: {np.min(vd_count_stats)} - {np.max(vd_count_stats)}")
        print(f"VD 數量標準差: {np.std(vd_count_stats):.1f}")
    
    if data_availability_stats:
        print(f"\n=== 資料可用性 ===")
        print(f"平均有資料的 VD 比例: {np.mean(data_availability_stats):.1%}")
        print(f"可用性範圍: {np.min(data_availability_stats):.1%} - {np.max(data_availability_stats):.1%}")
        print(f"可用性標準差: {np.std(data_availability_stats):.3f}")
    
    if error_code_stats:
        print(f"\n=== 錯誤碼統計 ===")
        for key, percentages in error_code_stats.items():
            if percentages:  # 確保列表不為空
                avg_percentage = np.mean(percentages)
                if avg_percentage > 0.1:  # 只顯示 >0.1% 的錯誤
                    min_pct = np.min(percentages)
                    max_pct = np.max(percentages)
                    print(f"{key}: 平均 {avg_percentage:.1%} (範圍: {min_pct:.1%}-{max_pct:.1%})")
    
    if processing_errors:
        print(f"\n=== 處理錯誤 ({len(processing_errors)}) ===")
        for error in processing_errors[:10]:  # 只顯示前10個錯誤
            print(f"• {error}")
        if len(processing_errors) > 10:
            print(f"... 還有 {len(processing_errors) - 10} 個錯誤")


def main():
    parser = argparse.ArgumentParser(description="交通資料品質檢查工具")
    parser.add_argument("--h5_file", type=str, help="H5 檔案路徑")
    parser.add_argument("--json_dir", type=str, help="JSON 目錄路徑")
    parser.add_argument("--sample_size", type=int, default=10, help="JSON 檢查的樣本大小")
    parser.add_argument("--all", action="store_true", help="檢查所有預設檔案")
    
    args = parser.parse_args()
    
    if args.all:
        # 檢查預設檔案
        default_h5 = "blob/dataset/pre-processed/h5/traffic_features.h5"
        default_json = "blob/dataset/pre-processed/unzip_to_json"
        
        if os.path.exists(default_h5):
            comprehensive_h5_quality_check(default_h5)
            print("\n" + "="*80 + "\n")
        
        if os.path.exists(default_json):
            check_json_quality(default_json, args.sample_size)
    
    else:
        if args.h5_file:
            comprehensive_h5_quality_check(args.h5_file)
        
        if args.json_dir:
            if args.h5_file:
                print("\n" + "="*80 + "\n")
            check_json_quality(args.json_dir, args.sample_size)
    
    if not (args.h5_file or args.json_dir or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()