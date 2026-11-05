import pandas as pd
import glob
import os
import numpy as np

# Ruta de tu dataset
dataset_path = "/home/kstro/proyectos/TesisBack/tools/dataset/dataset-35k-pokemon"

print(f"📂 Inspeccionando dataset en: {dataset_path}")
files = glob.glob(os.path.join(dataset_path, "*.parquet"))

if not files:
    print("❌ No se encontraron archivos .parquet")
else:
    f = files[0]
    print(f"📄 Leyendo archivo: {os.path.basename(f)}")
    
    try:
        # Leer el parquet con pandas
        df = pd.read_parquet(f)
        print(f"📊 Columnas encontradas: {df.columns.tolist()}")
        
        # 1. Verificar nombres exactos (El script exige minúsculas estrictas)
        required = {'filename', 'caption', 'encoding'}
        present = set(df.columns)
        missing = required - present
        
        if missing:
            print(f"❌ ERROR FATAL: Faltan columnas o tienen otro nombre: {missing}")
            print("   (El script 'compute_topk' exige exactamente: 'filename', 'caption', 'encoding')")
        else:
            print("✅ Nombres de columnas correctos.")
            
            # 2. Verificar el tipo y longitud del encoding
            sample_enc = df['encoding'].iloc[0]
            print(f"🔍 Tipo de 'encoding': {type(sample_enc)}")
            
            # Verificar si es array de numpy o lista
            enc_len = 0
            if hasattr(sample_enc, '__len__'):
                enc_len = len(sample_enc)
            elif hasattr(sample_enc, 'size'):
                enc_len = sample_enc.size
                
            print(f"📏 Longitud de 'encoding': {enc_len}")
            
            if enc_len != 256:
                print(f"⚠️ ¡ALERTA DE TAMAÑO! Tienes {enc_len} tokens, pero el script espera 256.")
                print("   -> SOLUCIÓN: Edita la línea 167 de compute_topk_per_neuron.py")
                print(f"   Cambia: if enc_tokens.size != 256:")
                print(f"   Por:    if enc_tokens.size != {enc_len}:")
            else:
                print("✅ Tamaño 256 correcto.")

            # 3. Verificar nulos
            if df['encoding'].isnull().any():
                print("⚠️ Hay filas con 'encoding' NULO/NaN. Esto podría causar saltos.")
            else:
                print("✅ No hay nulos en 'encoding'.")

    except Exception as e:
        print(f"💥 Error crítico leyendo el parquet: {e}")