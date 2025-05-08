import argparse
from koopman_predictor_clu_geo import KoopmanPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_len', type=int, default=12, help='예측 길이')
    parser.add_argument('--data_dir', type=str, default='lobby3', help='훈련 데이터가 있는 디렉터리')
    parser.add_argument('--pattern', type=str, default=r'^\d+\.npy$', help='훈련 데이터 파일 패턴')
    parser.add_argument('--model_path', type=str, default='koopman_model_clu_geo.pkl', help='저장할 모델 파일 경로')
    args = parser.parse_args()

    print("Koopman Predictor (Cluster + Geometry) 학습 중...")

    # 모델 인스턴스 생성 시,
    # 만약 model_path가 이미 존재한다면 자동으로 로드, 존재하지 않으면 data_dir로 학습 후 저장
    model = KoopmanPredictor(prediction_len=args.prediction_len,
                             data_dir=args.data_dir,
                             pattern=args.pattern,
                             model_file=args.model_path)

    # 추가로, 아래와 같이 강제로 저장할 수도 있음
    # (이미 __init__에서 한 번 저장이 진행되지만 필요시 호출 가능)
    # model.save_model(args.model_path)

    print(f"모델이 {args.model_path}에 준비되었습니다.")
