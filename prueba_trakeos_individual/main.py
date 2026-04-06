import os
import argparse
from tracker_core import TrackingSystem

def main():
    parser = argparse.ArgumentParser(description="Prueba individual de trackeos modulares.")
    parser.add_argument("video", type=str, help="Nombre del video a procesar, por ejemplo 'video5' o 'video5.mp4'")
    parser.add_argument("tracker", type=str, nargs='?', default="ALL", 
                        help="Modelo de tracking a usar (ej: strongsort). Si omites esto, se procesan los 5 modelos.")
    
    args = parser.parse_args()

    # Formatear el nombre del video
    video_name = args.video
    if not video_name.endswith('.mp4'):
        video_name += '.mp4'

    # Construir las rutas hacia la raíz del proyecto
    # El archivo actual está en 'prueba_trakeos_individual/main.py'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    video_path = os.path.join(base_dir, "videos_para_testear", video_name)
    model_path = os.path.join(base_dir, "FAL-zi_v1_DB-egana-v2_best.pt")
    
    if not os.path.exists(video_path):
        print(f"Error: No se encontró el video en {video_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    # Determinar qué modelos correr
    if args.tracker == "ALL":
        trackers_to_eval = ['strongsort', 'bytetrack', 'botsort', 'ocsort', 'deepocsort']
        print(f"Modo completo. Se probarán los {len(trackers_to_eval)} modelos en {video_name}.")
    else:
        trackers_to_eval = [args.tracker.lower()]
        print(f"Modo individual. Se probará únicamente [{trackers_to_eval[0]}] en {video_name}.")

    # Ejecutar evaluaciones
    for tracker_name in trackers_to_eval:
        print(f"\n================ Procesando con Tracker: {tracker_name} ================")
        
        try:
            sys = TrackingSystem(model_path=model_path, tracker_type=tracker_name)
        except Exception as e:
            print(f"Error inicializando {tracker_name}: {e}")
            continue
            
        # Generar nombre del video de salida en una carpeta dedicada
        # Ejemplo: resultados_video5/strongsortvideo5.mp4
        nombre_base_video = video_name.split('.')[0]
        carpeta_resultados = f"resultados_{nombre_base_video}"
        
        # Crea la carpeta si no existe
        os.makedirs(carpeta_resultados, exist_ok=True)
        
        out_path = os.path.join(carpeta_resultados, f"{tracker_name}{nombre_base_video}.mp4")
        
        # Procesar
        sys.reset()
        print(f"  -> Extrayendo trayectorias... Esto puede tardar si el video es pesado.")
        res = sys.process_source(video_path, save_path=out_path, show=False)
        
        if res:
            print(f"  --> Finalizado. Resultado guardado en: {out_path}")
            print(f"      Métricas: {res['total_detections']} detecciones, {res['unique_ids']} IDs únicos de prendas.")

if __name__ == '__main__':
    main()
