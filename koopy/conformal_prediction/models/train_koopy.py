import argparse
import time  # 추가
from koopy_predictor import KoopmanPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_len', type=int, default=12, help='예측 길이')
    parser.add_argument('--data_dir', type=str, default='lobby3', help='학습 데이터가 있는 디렉토리')
    parser.add_argument('--pattern', type=str, default=r'^\d+\.npy$', help='파일명 패턴')
    parser.add_argument('--model_path', type=str, default='koopy.pkl', help='모델 저장 파일 경로')
    parser.add_argument('--num_epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    print("[MAIN] Creating KoopmanPredictor model...")
    model = KoopmanPredictor(prediction_len=args.prediction_len,
                             data_dir=args.data_dir,
                             pattern=args.pattern,
                             model_file=args.model_path)

    if model.K is None:
        print("[MAIN] Start training because K is None or no saved model found.")
        start_time = time.time()
        model.train(num_epochs=args.num_epochs, lr=args.lr, data_dir=args.data_dir)
        elapsed_time = time.time() - start_time
        model.save_model(args.model_path)
    else:
        print("[MAIN] Model was already loaded. If you want to re-train, remove the pkl file or rename it.")

    print(f"[MAIN] Model is ready at {args.model_path}.")
    print(elapsed_time)
