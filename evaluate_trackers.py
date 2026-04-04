import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from system import TrackingSystem
import argparse

def evaluate(video_name=None):
    if video_name:
        # Se asegura de añadir la extensión si el usuario solo pone "video0"
        if not video_name.endswith('.mp4'):
            video_name += '.mp4'
        path = os.path.join("videos_para_testear", video_name)
        if not os.path.exists(path):
            print(f"Error: No se encontró el video {path}")
            return
        video_paths = [path]
    else:
        video_paths = sorted(glob.glob("videos_para_testear/*.mp4"))[:13]
        if not video_paths:
            print("Error: No se encontraron videos en 'videos_para_testear/'")
            return

    trackers_to_eval = ['strongsort', 'bytetrack', 'botsort', 'ocsort', 'deepocsort']
    model_path = 'FAL-zi_v1_DB-egana-v2_best.pt'
    reid_model = 'osnet_x0_25_msmt17.pt'

    results = []

    for tracker_name in trackers_to_eval:
        print(f"\n================ Eval Tracker: {tracker_name} ================")
        
        try:
            sys = TrackingSystem(model_path=model_path, tracker_type=tracker_name, reid_weights=reid_model)
        except Exception as e:
            print(f"Error inicializando {tracker_name}: {e}")
            continue
        
        for vp in video_paths:
            v_name = os.path.basename(vp)
            print(f"  -> Procesando {v_name}...")
            sys.reset()
            res = sys.process_source(vp, save_path=None, show=False)
            
            results.append({
                "Tracker": tracker_name,
                "Video": v_name,
                "Total_Detections": res["total_detections"],
                "Unique_IDs": res["unique_ids"],
                "FPS": res["avg_fps"]
            })

    if not results:
        print("No se obtuvieron resultados válidos.")
        return

    df = pd.DataFrame(results)

    # Análisis de los algoritmos
    tracker_stats = df.groupby('Tracker').agg(
        Avg_FPS=('FPS', 'mean'),
        Total_IDs_Sum=('Unique_IDs', 'sum')
    ).reset_index()

    best_tracker = tracker_stats.sort_values(by=['Total_IDs_Sum', 'Avg_FPS'], ascending=[True, False]).iloc[0]['Tracker']

    print("\n==== RESUMEN DE MÉTRICAS ====")
    print(tracker_stats.to_markdown(index=False))
    print(f"\nEl MEJOR algoritmo seleccionado automáticamente es: {best_tracker}")
    print(f"Razón: Generó en total el menor número de Unique IDs de todos (menor fragmentación de trayectorias e ID switching), con un buen balance de FPS.")

    # Al terminar, exportamos el video SÓLO con el algoritmo ganador.
    print(f"\nGenerando videos procesados y gráficos del mejor tracker: {best_tracker} ...")
    best_sys = TrackingSystem(model_path, tracker_type=best_tracker, reid_weights=reid_model)
    os.makedirs("resultados_videos", exist_ok=True)

    df_best = df[df['Tracker'] == best_tracker]

    for vp in video_paths:
        v_name = os.path.basename(vp)
        out_path = os.path.join("resultados_videos", f"{best_tracker}_{v_name}")
        best_sys.reset()
        best_sys.process_source(vp, save_path=out_path, show=False)
        print(f"  --> Guardado video con trayectoria anotada: {out_path}")

    # Generar gráfico comparativo. (Solo util si evalúa múltiples videos, pero lo generamos de todos modos)
    plt.figure(figsize=(12, 6))
    x = range(len(df_best))
    width = 0.35

    plt.bar([i - width/2 for i in x], df_best['Total_Detections'], width, label='Detecciones Totales (Frames)', color='skyblue')
    plt.bar([i + width/2 for i in x], df_best['Unique_IDs'], width, label='IDs Únicas Generadas', color='darkblue')

    plt.title(f'Rendimiento en Tracking ({best_tracker}) - Comparación Detecciones vs IDs')
    plt.ylabel('Cantidad')
    plt.xlabel('Video')
    plt.xticks(x, df_best['Video'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plot_path = "reporte_metricas_mejor_tracker.png" if not video_name else f"reporte_metricas_{best_tracker}_{video_name.split('.')[0]}.png"
    plt.savefig(plot_path)
    print(f"\nGráfico guardado en: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluar Trackers sobre videos de Falabella")
    parser.add_argument("video", nargs='?', type=str, default=None, 
                        help="Nombre del video a procesar (ej. 'video0' o 'video0.mp4'). Si lo omites, evaluará los 13 videos.")
    args = parser.parse_args()
    
    evaluate(args.video)
