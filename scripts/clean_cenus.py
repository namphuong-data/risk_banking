import pandas as pd

df = pd.read_csv("/home/namphuong/Desktop/DE_banking/data/zip_geo_final.csv")

# Loại bỏ lãnh thổ hải ngoại không có trong Lending Club
EXCLUDE_STATES = ["PR", "VI", "GU", "AS", "MP"]
df = df[~df["state_id"].isin(EXCLUDE_STATES)].reset_index(drop=True)

# Loại bỏ dòng area_type null
df = df[df["area_type"] != "nan"].reset_index(drop=True)

print(f"Sau khi lọc: {len(df):,} zip_prefix")
print(df["state_id"].nunique(), "bang")

df.to_csv(
    "/home/namphuong/Desktop/DE_banking/data/zip_geo_final.csv",
    index=False,
    encoding="utf-8-sig",
)
print("[✓] Đã lưu lại file sạch")
