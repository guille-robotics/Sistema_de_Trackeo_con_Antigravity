import argparse
from system import TrackingSystem

def run_production_stream(rtsp_url, model_path, best_tracker='botsort', reid_weights='osnet_x0_25_msmt17.pt'):
    print(f"Iniciando tracker en STREAMING con {best_tracker}...")
    sys = TrackingSystem(model_path=model_path, tracker_type=best_tracker, reid_weights=reid_weights)
    
    # Process_source soporta rtsp nativamente. show=True para tiempo real.
    # En un entorno headless (producción server) poner show=False o redirigir el cv2_imshow a un frontend streaming web.
    sys.process_source(rtsp_url, save_path=None, show=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP Stream URL")
    parser.add_argument("--tracker", type=str, default="botsort", help="Mejor tracker elegido")
    args = parser.parse_args()
    
    run_production_stream(
        rtsp_url=args.rtsp,
        model_path="FAL-zi_v1_DB-egana-v2_best.pt",
        best_tracker=args.tracker
    )
