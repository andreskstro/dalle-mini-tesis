import pandas as pd
import glob

# Revisa uno de tus parquets modificados
file = glob.glob("001.parquet")[0]
df = pd.read_parquet(file)

# Buscar la palabra prohibida en el TEXTO visible
cerdos = df[df['caption'].str.contains(" snake ", case=False)]

if len(cerdos) > 0:
    print(f"❌ ¡Alerta! Todavía hay {len(cerdos)} cerdos en el archivo.")
    print(cerdos['caption'].iloc[0])
else:
    print("✅ ¡Limpio! No hay cerdos en los captions.")
    print("🚀 PUEDES INICIAR EL ENTRENAMIENTO YA.")