#!/usr/bin/env python3
"""
GSOD 多站点数据整理脚本 (方案 A：按最新日期筛选)

用途：
- 读取多个 GSOD 站点 CSV 文件（每个文件一个站点，可能包含多日期）
- 只保留最新日期的数据
- 按站点聚合
- 清洗异常值（温度 > 60°C 或 < -50°C 视为异常）
- 输出为 parquet 格式（供 run_kriging.py 使用）

支持目标变量选择：温度（temperature）或露点温度（dew_point）

使用示例：
    python run_prepare_gsod.py                          # 使用默认路径，直接覆盖
    python run_prepare_gsod.py --input-dir ./other_data # 指定自定义路径
    python run_prepare_gsod.py --no-latest-date         # 保留所有日期数据
    python run_prepare_gsod.py --target dew_point       # 使用露点温度作为主目标
    python run_prepare_gsod.py --out my_data.parquet    # 指定输出文件
"""
import argparse
import os
import sys

# ============================================================================
# 🔧 配置区：修改这里的默认路径
# ============================================================================

# GSOD CSV 文件所在目录（相对于脚本位置或绝对路径）
DEFAULT_INPUT_DIR = './2024_gsod_data'

# 输出文件名
DEFAULT_OUTPUT_FILE = 'gsod_merged_cache.parquet'

# 文件匹配模式（支持 glob 通配符）
DEFAULT_FILE_PATTERN = '*.csv'

# 温度单位：'F'(华氏度) 或 'C'(摄氏度)，GSOD 标准为 'F'
DEFAULT_TEMP_UNIT = 'F'

# 是否只保留最新日期的数据（推荐 True）
DEFAULT_USE_LATEST_DATE = True

# 默认目标变量：'temperature' 或 'dew_point'
DEFAULT_TARGET = 'temperature'

# 异常值过滤阈值（摄氏度）
DEFAULT_TEMP_MIN = -50.0   # 低于此值视为异常
DEFAULT_TEMP_MAX = 60.0    # 高于此值视为异常（GSOD 全球最高温约 56.7°C）


# ============================================================================
# ============================================================================


def main():
    """主函数"""
    from kriging_rf.gsod_utils import read_gsod_directory, prepare_gsod

    parser = argparse.ArgumentParser(
        description="准备 GSOD 站点数据用于 kriging_rf pipeline (方案 A：最新日期)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认路径，只保留最新日期（推荐）
  python run_prepare_gsod.py

  # 指定自定义输入目录
  python run_prepare_gsod.py --input-dir /path/to/gsod/files

  # 保留所有日期数据（会得到更多样本）
  python run_prepare_gsod.py --no-latest-date

  # 使用露点温度作为目标变量
  python run_prepare_gsod.py --target dew_point --out gsod_dewp.parquet

  # 完整示例
  python run_prepare_gsod.py \\
    --input-dir ./GSOD_data \\
    --pattern "station_*.csv" \\
    --temp-unit F \\
    --target dew_point \\
    --out gsod_final.parquet \\
    --use-latest-date
        """
    )

    parser.add_argument(
        '--input-dir', '-d',
        default=DEFAULT_INPUT_DIR,
        help=f'GSOD CSV 文件所在目录 (默认: {DEFAULT_INPUT_DIR})'
    )

    parser.add_argument(
        '--pattern', '-p',
        default=DEFAULT_FILE_PATTERN,
        help=f'文件匹配模式 (默认: {DEFAULT_FILE_PATTERN})'
    )

    parser.add_argument(
        '--temp-unit',
        choices=['F', 'C'],
        default=DEFAULT_TEMP_UNIT,
        help=f'温度单位：F(华氏度) 或 C(摄氏度) (默认: {DEFAULT_TEMP_UNIT})'
    )

    parser.add_argument(
        '--target',
        choices=['temperature', 'dew_point'],
        default=DEFAULT_TARGET,
        help=f'目标变量：temperature（温度）或 dew_point（露点温度）(默认: {DEFAULT_TARGET})'
    )

    parser.add_argument(
        '--out', '-o',
        default=DEFAULT_OUTPUT_FILE,
        help=f'输出 parquet 文件路径 (默认: {DEFAULT_OUTPUT_FILE})'
    )

    parser.add_argument(
        '--use-latest-date',
        action='store_true',
        default=DEFAULT_USE_LATEST_DATE,
        help=f'只保留最新日期的数据 (默认: {DEFAULT_USE_LATEST_DATE})'
    )

    parser.add_argument(
        '--no-latest-date',
        action='store_false',
        dest='use_latest_date',
        help='保留所有日期的数据（不筛选）'
    )

    # 异常值过滤参数
    parser.add_argument(
        '--temp-min',
        type=float,
        default=DEFAULT_TEMP_MIN,
        help=f'温度最小阈值（摄氏度），低于此值视为异常 (默认: {DEFAULT_TEMP_MIN})'
    )
    parser.add_argument(
        '--temp-max',
        type=float,
        default=DEFAULT_TEMP_MAX,
        help=f'温度最大阈值（摄氏度），高于此值视为异常 (默认: {DEFAULT_TEMP_MAX})'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GSOD 数据预处理 (方案 A：最新日期筛选)")
    print("=" * 70)
    print(f"\n📋 配置参数：")
    print(f"  输入目录：{args.input_dir}")
    print(f"  文件模式：{args.pattern}")
    print(f"  温度单位：{args.temp_unit}")
    print(f"  目标变量：{args.target}")
    print(f"  只保留最新日期：{args.use_latest_date}")
    print(f"  输出文件：{args.out}")
    print(f"  温度过滤范围：{args.temp_min} ~ {args.temp_max}°C")

    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"\n❌ 错误：输入目录不存在：{args.input_dir}")
        print(f"\n💡 提示：")
        print(f"   1. 检查目录路径是否正确")
        print(f"   2. 在脚本中修改 DEFAULT_INPUT_DIR 变量")
        print(f"   3. 或使用 --input-dir 参数指定路径")
        sys.exit(1)

    try:
        # 读取所有 GSOD CSV
        print(f"\n📂 从目录读取 GSOD CSV 文件...")
        df = read_gsod_directory(args.input_dir, pattern=args.pattern)

        print(f"\n🔄 数据处理中（温度单位：{args.temp_unit}，目标：{args.target}）...")
        grouped = prepare_gsod(
            df,
            temp_unit=args.temp_unit,
            drop_nan_temp=True,
            use_latest_date=args.use_latest_date,
            target_col=args.target,
            temp_min=args.temp_min,
            temp_max=args.temp_max
        )

        print(f"\n✅ 成功聚合至 {len(grouped)} 个站点")
        print("\n📊 示例数据：")
        print(grouped.head(15).to_string(index=False))

        # 统计信息
        print(f"\n📈 数据统计：")
        print(f"  站点总数：{len(grouped)}")
        print(f"  温度范围：{grouped['temperature'].min():.2f}°C ~ {grouped['temperature'].max():.2f}°C")
        if 'dew_point' in grouped.columns and not grouped['dew_point'].isna().all():
            print(f"  露点温度范围：{grouped['dew_point'].min():.2f}°C ~ {grouped['dew_point'].max():.2f}°C")
        print(f"  海拔范围：{grouped['elevation'].min():.0f}m ~ {grouped['elevation'].max():.0f}m")

        # 默认覆盖
        if os.path.exists(args.out):
            print(f"\n⚠️  输出文件已存在，将覆盖：{args.out}")

        # ========== 关键修复：确保列类型正确 ==========
        # station_id 可能包含字母，强制转为字符串
        grouped['station_id'] = grouped['station_id'].astype(str)
        # name 列可能包含非字符串
        if 'name' in grouped.columns:
            grouped['name'] = grouped['name'].astype(str)
        # =============================================

        # 保存为 parquet（优先）
        try:
            import pandas as pd
            grouped.to_parquet(args.out, index=False, compression='snappy')
            file_size_mb = os.path.getsize(args.out) / (1024 ** 2)
            print(f"\n✅ 已保存合并缓存到：{args.out}")
            print(f"   文件大小：{file_size_mb:.2f} MB")
            print(f"\n💡 后续使用 kriging 流程：")
            print(f"   python run_kriging.py --cache {args.out} --target {args.target} --n-splits 5 --ok-neighbors 15")

        except Exception as e:
            print(f"\n⚠️  Parquet 保存失败：{e}")
            # 退回保存为 CSV
            out_csv = os.path.splitext(args.out)[0] + ".csv"
            grouped.to_csv(out_csv, index=False)
            print(f"✅ 已保存为 CSV 格式：{out_csv}")

    except Exception as e:
        print(f"\n❌ 处理出错：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()