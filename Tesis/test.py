import pandas as pd

csv_path = r"C:\Users\Andrés Castro\Desktop\data\captions.csv"
df = pd.read_csv(csv_path)

voltorb = df[df["image"].astype(str).str.startswith("Voltorb/")].copy()
print("Filas Voltorb en CSV:", len(voltorb))
print(voltorb["image"].head(20).to_string(index=False))