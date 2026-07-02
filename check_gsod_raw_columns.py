"""
检查原始 GSOD 数据文件夹中 CSV 文件的列名
"""
import pandas as pd
import os
import glob

# ====== 请修改为你的 GSOD 原始数据文件夹路径 ======
GSOD_FOLDER = r"D:\pycharm\Kriging_test\Kriging and Machine Learning\2024_gsod_data"


# ================================================

def check_raw_gsod(folder_path):
    # 获取所有 CSV 文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"❌ 未在 {folder_path} 中找到 CSV 文件")
        return

    print(f"✅ 找到 {len(csv_files)} 个 CSV 文件")

    # 读取第一个文件
    sample_file = csv_files[0]
    print(f"\n📄 读取示例文件: {os.path.basename(sample_file)}")

    try:
        df = pd.read_csv(sample_file)

        print("\n" + "=" * 60)
        print("📋 全部列名（原始 GSOD 格式）：")
        print("=" * 60)
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        print("\n" + "=" * 60)
        print("👀 前 3 行数据预览：")
        print("=" * 60)
        print(df.head(3).to_string())

        print("\n" + "=" * 60)
        print("🔍 湿度相关列检查：")
        print("=" * 60)

        # GSOD 中湿度相关的标准列名
        humidity_columns = {
            'DEWP': '露点温度 (°C)',
            'DEWP_ATTRIBUTES': '露点温度属性',
            'RH': '相对湿度 (%)',  # 部分 GSOD 包含
            'RELH': '相对湿度 (%)',
            'HUMIDITY': '湿度'
        }

        found = []
        for col in humidity_columns:
            if col in df.columns:
                found.append(col)
                non_null = df[col].notna().sum()
                print(f"  ✅ {col} ({humidity_columns[col]})")
                print(f"     非空值: {non_null}/{len(df)}")
                if non_null > 0 and df[col].dtype in ['float64', 'int64']:
                    print(f"     范围: {df[col].min():.4f} ~ {df[col].max():.4f}")
                    print(f"     均值: {df[col].mean():.4f}")
            else:
                print(f"  ❌ {col} 不存在")

        if not found:
            print("\n⚠️ 未在示例文件中发现湿度相关列。")
            print("💡 提示：可以检查其他文件，或确认数据是否包含湿度字段。")
            print("📌 注意：DEWP（露点温度）是 GSOD 标准数据集中最常见的湿度指标。")

        # 打印温度列信息
        print("\n" + "=" * 60)
        print("🌡️ 温度列检查：")
        print("=" * 60)
        if 'TEMP' in df.columns:
            temp_mean = df['TEMP'].mean() / 10  # GSOD 温度单位是 0.1°C
            print(f"  ✅ TEMP 存在，平均值 (已除以10): {temp_mean:.2f}°C")
        else:
            print("  ❌ TEMP 列不存在（可能已做其他处理）")

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")


if __name__ == "__main__":
    check_raw_gsod(GSOD_FOLDER)