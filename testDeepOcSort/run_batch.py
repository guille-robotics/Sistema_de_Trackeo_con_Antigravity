import os
import pandas as pd
from tracker_deepocsort import DeepOCSortTrackingSystem

def main():
    videos_a_probar = [f'video{i}.mp4' for i in range(14)]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "FAL-zi_v1_DB-egana-v2_best.pt")
    
    print(f"\n================ Iniciando Procesamiento Batch de DeepOC-SORT ================")
    
    sys = DeepOCSortTrackingSystem(model_path=model_path, reid_weights='osnet_x0_25_msmt17.pt')
    
    carpeta_resultados = "resultados_thr_correcto"
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    # Archivo para guardar resumen
    log_file = os.path.join(carpeta_resultados, "metricas_batch.txt")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n================ NUEVA EJECUCIÓN BATCH ================\n")
        
    resultados_excel = []
    
    for vid in videos_a_probar:
        video_path = os.path.join(base_dir, "videos_para_testear", vid)
        if not os.path.exists(video_path):
            print(f"Saltando {vid}, no existe.")
            continue
            
        print(f"\n--->  Procesando video: {vid}")
        out_path = os.path.join(carpeta_resultados, f"DeepOCSORT_{vid}")
        
        sys.reset()
        res = sys.process_source(video_path, save_path=out_path, show=False)
        
        if res:
            res_str = (
                f"Video: {vid} | "
                f"Detecciones Totales: {res['total_detections']} | "
                f"IDs Únicos: {res['unique_ids']} | "
                f"FPS: {res['avg_fps']:.2f}\n"
            )
            print(f"  --- {res_str.strip()} ---")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(res_str)
                
            # Guardar resultados para el Excel
            resultados_excel.append({
                'Video': vid,
                'IDs Totales': res['unique_ids'],
                'IDs por Clase': str(res['unique_ids_per_class']),
                'Tiempo Inferencia RTDETR Total (s)': res['total_rtdetr_time'],
                'Tiempo Inferencia Tracking Total (s)': res['total_tracking_time'],
                'Tiempo Flujo Total (s)': res['total_pipeline_time'],
                'Tiempo Inferencia RTDETR Promedio (s)': res['avg_rtdetr_time'],
                'Tiempo Inferencia Tracking Promedio (s)': res['avg_tracking_time'],
                'Tiempo Tracking Flujo Total Promedio (s)': res['avg_pipeline_time'],
                'FPS Promedio': res['avg_fps']
            })

    # Guardar a excel
    if resultados_excel:
        df = pd.DataFrame(resultados_excel)
        excel_path = os.path.join(carpeta_resultados, "metricas_trackeo.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nSe generó exitosamente el archivo Excel en: {excel_path}")

    print("\n================ PROCESAMIENTO BATCH TERMINADO ================")
    print(f"Los videos y en el archivo {log_file} están guardados en {carpeta_resultados}.")

if __name__ == '__main__':
    main()
