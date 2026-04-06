import os
import argparse
from tracker_deepocsort import DeepOCSortTrackingSystem

def main():
    parser = argparse.ArgumentParser(description="Prueba individual de DeepOC-SORT.")
    parser.add_argument("video", type=str, nargs='?', default="video5.mp4", help="Nombre del video a procesar (ej: video5.mp4)")
    
    args = parser.parse_args()

    video_name = args.video
    if not video_name.endswith('.mp4'):
        video_name += '.mp4'

    # Rutas relativas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(base_dir, "videos_para_testear", video_name)
    model_path = os.path.join(base_dir, "FAL-zi_v1_DB-egana-v2_best.pt")
    
    if not os.path.exists(video_path):
        print(f"Error: No se encontró el video en {video_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo híbrido RT-DETR en {model_path}")
        return

    print(f"\n================ Iniciando Prueba de DeepOC-SORT con {video_name} ================")
    
    # Inicializando
    sys = DeepOCSortTrackingSystem(model_path=model_path, reid_weights='osnet_x0_25_msmt17.pt')
    
    # Carpeta resultados local
    carpeta_resultados = "resultados_deepocsort"
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    out_path = os.path.join(carpeta_resultados, f"DeepOCSORT_{video_name}")
    
    print(f"  -> Extrayendo trayectorias con YOLOX/RTDert + DeepOC-SORT ReID. Por favor espera...")
    res = sys.process_source(video_path, save_path=out_path, show=False)
    
    if res:
        print(f"  --> Finalizado de Procesar.\n  --> Resultado en video guardado en: {out_path}")
        print(f"  --> DeepOC-SORT Métricas Generadas:")
        print(f"      * Detecciones Totales: {res['total_detections']}")
        print(f"      * Prendas Únicas Asignadas (IDs): {res['unique_ids']}")
        print(f"      * FPS Promedio: {res['avg_fps']:.2f}")

if __name__ == '__main__':
    main()
