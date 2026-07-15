"""
load_data_to_postgres.py
────────────────────────
Load toàn bộ data nguồn vào PostgreSQL cho project Credit Risk Analytics.

Các bảng được tạo:
  - accepted_loans   : 28M dòng từ accepted_2007_to_2018Q4.csv
  - rejected_loans   : 2.2M dòng từ rejected_2007_to_2018Q4.csv
  - fred_macro       : 156 dòng từ fred_macro_2007_2018.csv

Cách chạy:
    pip install sqlalchemy psycopg2-binary pandas pyarrow tqdm
    python load_data_to_postgres.py

Cấu hình kết nối và đường dẫn file ở phần CONFIG bên dưới.
"""

import os
import sys
import time
import io
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from tqdm import tqdm

# ── CONFIG — chỉnh sửa tại đây ────────────────────────────────────────────────

PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "risk_banking"
PG_USER = "namphuong"
PG_PASSWORD = "2104"

DATA_DIR = "/home/namphuong/Desktop/DE_banking/data"

FILE_ACCEPTED = os.path.join(DATA_DIR, "accepted_2007_to_2018Q4.csv")
FILE_REJECTED = os.path.join(DATA_DIR, "rejected_2007_to_2018Q4.csv")
FILE_MACRO = os.path.join(DATA_DIR, "fred_macro_2007_2018.csv")

CHUNK_SIZE = 50_000  # số dòng mỗi lần đọc từ CSV

# ── Kết nối psycopg2 trực tiếp để dùng COPY (nhanh hơn to_sql 10-20x) ────────


def get_psycopg2_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def copy_from_df(df: pd.DataFrame, table: str, conn):
    """Insert DataFrame vào PostgreSQL bằng COPY — nhanh nhất có thể."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({','.join(df.columns)}) " f"FROM STDIN WITH CSV NULL '\\N'",
            buffer,
        )
    conn.commit()


# ── Cột cần thiết từ accepted_loans ───────────────────────────────────────────

ACCEPTED_COLS = [
    # Định danh & thời gian
    "id",
    "issue_d",
    # Thông tin khoản vay
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "purpose",
    # Thông tin khách hàng
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "addr_state",
    "zip_code",
    "dti",
    # Trạng thái khoản vay — cốt lõi phân loại nợ
    "loan_status",
    # Thanh toán và thu hồi — dùng tính LGD / EAD
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    # Lịch sử tín dụng — feature cho PD model
    "fico_range_low",
    "fico_range_high",
    "last_fico_range_low",
    "last_fico_range_high",
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    # Loại hồ sơ
    "application_type",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_engine():
    url = (
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    )
    return create_engine(url, pool_pre_ping=True)


def check_files():
    """Kiểm tra các file nguồn tồn tại trước khi chạy."""
    missing = []
    for path in [FILE_ACCEPTED, FILE_REJECTED, FILE_MACRO]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("\n[LỖI] Không tìm thấy file:")
        for f in missing:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("[✓] Tất cả file nguồn tồn tại")


def check_connection(engine):
    """Kiểm tra kết nối PostgreSQL."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[✓] Kết nối PostgreSQL thành công — {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    except Exception as e:
        print(f"\n[LỖI] Không kết nối được PostgreSQL: {e}")
        print("      Kiểm tra lại HOST, PORT, USER, PASSWORD trong CONFIG")
        sys.exit(1)


def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def estimate_chunks(path, chunk_size):
    """Ước tính số chunk dựa trên số dòng file."""
    try:
        with open(path) as f:
            n_lines = sum(1 for _ in f) - 1  # trừ header
        return max(1, n_lines // chunk_size + 1), n_lines
    except Exception:
        return None, None


# ── Load accepted_loans ───────────────────────────────────────────────────────


def transform_accepted(chunk: pd.DataFrame) -> pd.DataFrame:
    """Transform chunk accepted_loans trước khi insert."""

    # Parse ngày phát hành khoản vay
    chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    chunk["issue_month"] = chunk["issue_d"].dt.to_period("M").astype(str)

    # Phân loại nợ 5 nhóm theo TT11/2021
    status_map = {
        "Current": 1,
        "Fully Paid": 1,
        "In Grace Period": 2,
        "Late (16-30 days)": 2,
        "Late (31-120 days)": 3,
        "Default": 4,
        "Charged Off": 5,
    }
    chunk["debt_group"] = chunk["loan_status"].map(status_map).fillna(1).astype(int)

    # Xử lý null có ý nghĩa nghiệp vụ
    # null = chưa bao giờ bị delinquent → gán giá trị lớn
    chunk["mths_since_last_delinq"] = chunk["mths_since_last_delinq"].fillna(999)
    chunk["ever_delinquent"] = (chunk["mths_since_last_delinq"] < 999).astype(int)

    # Chuẩn hóa int_rate — bỏ dấu % nếu có
    if chunk["int_rate"].dtype == object:
        chunk["int_rate"] = (
            chunk["int_rate"].str.replace("%", "", regex=False).astype(float)
        )

    # Chuẩn hóa revol_util
    if chunk["revol_util"].dtype == object:
        chunk["revol_util"] = (
            chunk["revol_util"]
            .str.replace("%", "", regex=False)
            .astype(float, errors="ignore")
        )

    # Chuẩn hóa term — bỏ " months"
    if chunk["term"].dtype == object:
        chunk["term"] = (
            chunk["term"]
            .str.strip()
            .str.replace(" months", "", regex=False)
            .astype(float, errors="ignore")
        )

    # Chuẩn hóa emp_length
    emp_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
    chunk["emp_length_num"] = chunk["emp_length"].map(emp_map)

    return chunk


def create_accepted_table(engine, sample_df: pd.DataFrame):
    with engine.begin() as conn:  # begin() tự commit khi ra khỏi with
        conn.execute(text("DROP TABLE IF EXISTS accepted_loans;"))

    sample_df.head(0).to_sql(
        "accepted_loans",
        engine,
        if_exists="replace",
        index=False,
    )
    print("  [✓] Bảng accepted_loans đã được tạo")


def load_accepted(engine):
    print("\n" + "─" * 60)
    print("[1/3] Loading accepted_loans")
    print(f"      File : {FILE_ACCEPTED}")
    print(f"      Size : {get_file_size_mb(FILE_ACCEPTED):.0f} MB")

    n_chunks, n_rows = estimate_chunks(FILE_ACCEPTED, CHUNK_SIZE)
    if n_rows:
        print(f"      Rows : ~{n_rows:,}  |  Chunks: ~{n_chunks}")
    print("─" * 60)

    t0 = time.time()
    total_loaded = 0
    first_chunk = True
    pg_conn = None

    reader = pd.read_csv(
        FILE_ACCEPTED,
        usecols=lambda c: c in ACCEPTED_COLS,
        low_memory=False,
        dtype={"id": str, "zip_code": str},
        chunksize=CHUNK_SIZE,
    )

    with tqdm(total=n_chunks, unit="chunk", desc="  Inserting") as pbar:
        for chunk in reader:
            chunk = transform_accepted(chunk)

            if first_chunk:
                # Tạo bảng từ schema chunk đầu tiên
                create_accepted_table(engine, chunk)
                pg_conn = get_psycopg2_conn()
                first_chunk = False

            # Dùng COPY thay vì INSERT — nhanh hơn 10-20x
            copy_from_df(chunk, "accepted_loans", pg_conn)

            total_loaded += len(chunk)
            pbar.update(1)
            pbar.set_postfix({"rows": f"{total_loaded:,}"})

    if pg_conn:
        pg_conn.close()

    duration = time.time() - t0
    print(f"\n  [✓] {total_loaded:,} dòng — {format_duration(duration)}")
    return total_loaded


# ── Load rejected_loans ───────────────────────────────────────────────────────


def transform_rejected(df: pd.DataFrame) -> pd.DataFrame:
    """Transform rejected_loans — chuẩn hóa tên cột và kiểu dữ liệu."""

    # Chuẩn hóa tên cột
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Parse ngày
    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
        df["application_month"] = df["application_date"].dt.to_period("M").astype(str)

    # Chuẩn hóa debt_to_income_ratio — bỏ dấu %
    dti_col = "debt_to_income_ratio"
    if dti_col in df.columns and df[dti_col].dtype == object:
        df[dti_col] = (
            df[dti_col].str.replace("%", "", regex=False).astype(float, errors="ignore")
        )

    return df


def load_rejected(engine):
    print("\n" + "─" * 60)
    print("[2/3] Loading rejected_loans")
    print(f"      File : {FILE_REJECTED}")
    print(f"      Size : {get_file_size_mb(FILE_REJECTED):.0f} MB")

    n_chunks, n_rows = estimate_chunks(FILE_REJECTED, CHUNK_SIZE)
    if n_rows:
        print(f"      Rows : ~{n_rows:,}  |  Chunks: ~{n_chunks}")
    print("─" * 60)

    t0 = time.time()
    total_loaded = 0
    first_chunk = True
    pg_conn = None

    reader = pd.read_csv(
        FILE_REJECTED,
        low_memory=False,
        chunksize=CHUNK_SIZE,  # đọc từng chunk thay vì load toàn bộ
    )

    with tqdm(total=n_chunks, unit="chunk", desc="  Inserting") as pbar:
        for chunk in reader:
            chunk = transform_rejected(chunk)

            if first_chunk:
                # Tạo bảng từ schema chunk đầu tiên
                chunk.head(0).to_sql(
                    "rejected_loans",
                    engine,
                    if_exists="replace",
                    index=False,
                )
                pg_conn = get_psycopg2_conn()
                first_chunk = False

            # Insert bằng COPY
            copy_from_df(chunk, "rejected_loans", pg_conn)

            total_loaded += len(chunk)
            pbar.update(1)
            pbar.set_postfix({"rows": f"{total_loaded:,}"})

    if pg_conn:
        pg_conn.close()

    duration = time.time() - t0
    print(f"\n  [✓] {total_loaded:,} dòng — {format_duration(duration)}")
    return total_loaded


# ── Load fred_macro ───────────────────────────────────────────────────────────


def load_macro(engine):
    print("\n" + "─" * 60)
    print("[3/3] Loading fred_macro")
    print(f"      File : {FILE_MACRO}")
    print("─" * 60)

    t0 = time.time()

    df = pd.read_csv(FILE_MACRO)
    df.to_sql(
        "fred_macro",
        engine,
        if_exists="replace",
        index=False,
    )

    duration = time.time() - t0
    print(f"  [✓] {len(df):,} dòng — {format_duration(duration)}")
    return len(df)


# ── Tạo index sau khi load xong ──────────────────────────────────────────────


def create_indexes(engine):
    print("\n" + "─" * 60)
    print("[4/4] Tạo index tối ưu query ...")
    print("─" * 60)

    indexes = [
        ("idx_accepted_issue_month", "accepted_loans", "issue_month"),
        ("idx_accepted_loan_status", "accepted_loans", "loan_status"),
        ("idx_accepted_debt_group", "accepted_loans", "debt_group"),
        ("idx_accepted_grade", "accepted_loans", "grade"),
        ("idx_accepted_addr_state", "accepted_loans", "addr_state"),
        ("idx_accepted_purpose", "accepted_loans", "purpose"),
        ("idx_rejected_app_month", "rejected_loans", "application_month"),
        ("idx_rejected_state", "rejected_loans", "state"),
        ("idx_macro_issue_month", "fred_macro", "issue_month"),
    ]

    for idx_name, table, column in indexes:
        try:
            with engine.begin() as conn:  # begin() tự commit sau mỗi index
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({column});"
                    )
                )
            print(f"  [✓] {idx_name}")
        except Exception as e:
            print(f"  [!] {idx_name} — bỏ qua: {e}")


# ── Verify sau khi load ───────────────────────────────────────────────────────


def verify(engine):
    print("\n" + "─" * 60)
    print("[Verify] Kiểm tra dữ liệu đã load ...")
    print("─" * 60)

    queries = {
        "accepted_loans — tổng dòng": "SELECT COUNT(*) FROM accepted_loans",
        "accepted_loans — phân phối debt_group": """SELECT debt_group, COUNT(*) AS cnt,
               ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
               FROM accepted_loans GROUP BY debt_group ORDER BY debt_group""",
        "accepted_loans — khoảng thời gian": "SELECT MIN(issue_month), MAX(issue_month) FROM accepted_loans",
        "rejected_loans — tổng dòng": "SELECT COUNT(*) FROM rejected_loans",
        "fred_macro — khoảng thời gian": "SELECT MIN(issue_month), MAX(issue_month), COUNT(*) FROM fred_macro",
    }

    with engine.connect() as conn:
        for label, sql in queries.items():
            try:
                result = pd.read_sql(text(sql), conn)
                print(f"\n  {label}:")
                print(result.to_string(index=False))
            except Exception as e:
                print(f"\n  [LỖI] {label}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Credit Risk — Load Data to PostgreSQL")
    print(f"  Target: {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    print("=" * 60)

    check_files()
    engine = make_engine()
    check_connection(engine)

    t_start = time.time()

    rows_accepted = load_accepted(engine)
    rows_rejected = load_rejected(engine)
    rows_macro = load_macro(engine)
    create_indexes(engine)
    verify(engine)

    total_duration = time.time() - t_start

    print("\n" + "=" * 60)
    print("  HOÀN THÀNH")
    print(f"  accepted_loans : {rows_accepted:>12,} dòng")
    print(f"  rejected_loans : {rows_rejected:>12,} dòng")
    print(f"  fred_macro     : {rows_macro:>12,} dòng")
    print(f"  Tổng thời gian : {format_duration(total_duration)}")
    print("=" * 60)
    print("\n  Mở DBeaver → kết nối PostgreSQL → kiểm tra các bảng trên.")


if __name__ == "__main__":
    main()
