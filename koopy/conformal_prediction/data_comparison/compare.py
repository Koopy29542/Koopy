import numpy as np
import os

def compute_ADE_FDE(predictions, targets):
    """
    predictions와 targets의 shape는 (N, T, M, 2)를 가정.
    N: (시퀀스 수) 876
    T: (타임스텝) 12
    M: (에이전트 수) 44
    2: (x, y 좌표)
    """
    # 결괏값을 담을 리스트
    ade_list = []
    fde_list = []
    quater=0
    # 모든 시퀀스(N), 에이전트(M)에 대해
    for i in range(targets.shape[0]):
        for k in range(targets.shape[2]):
            pred_seq = predictions[i, :, k, :]  # (12, 2)
            tgt_seq = targets[i, :, k, :]       # (12, 2)
            """             if ~(np.isnan(tgt_seq).any()):
                if np.isnan(pred_seq).any():
                    quater=quater+1
                    print("AGENT")
                    print(k)
                    print("TIME FRAME")
                    print(i) """
            # 예측값 혹은 타겟에 NaN이 하나라도 있으면 스킵
            if np.isnan(pred_seq).any() or np.isnan(tgt_seq).any():
                #if not(np.isnan(pred_seq).all() or np.isnan(tgt_seq).all()):
                    #print("HELLO")
                    #print(tgt_seq)
                    #print(i)
                continue

                

            # 타임스텝별 L2 거리 계산 (shape: (12,))

            dist = np.linalg.norm(pred_seq - tgt_seq, axis=1)  

            # ADE: 12개 전체 거리의 평균
            ade_list.append(dist.mean())
            # FDE: 마지막 타임스텝(12번째) 거리
            fde_list.append(dist[-1])

    # 리스트가 비어있지 않다면 평균, 비어있으면 np.nan
    ADE = np.mean(ade_list) if len(ade_list) > 0 else np.nan
    FDE = np.mean(fde_list) if len(fde_list) > 0 else np.nan
    #print(len(ade_list))
    #print(quater)
    return ADE, FDE


if __name__ == "__main__":
    test_dirpaths = ['../lobby2/biwi_eth/test','../lobby2/univ/test','../lobby2/univ/test','../lobby2/biwi_hotel/test','../lobby2/crowds_zara01/test','../lobby2/crowds_zara02/test']
    directories=  ['biwi_eth','students003','students001','biwi_hotel','crowds_zara01','crowds_zara02']

    for i ,j in zip(test_dirpaths,directories):
        directory = i
    
        # 타겟 데이터 로드
        target_file = os.path.join(directory, j+"_targets.npy")
        targets = np.load(target_file)  # shape: (876, 12, 44, 2)

        # 예측 파일들 (.npy) 중에서 targets와 biwi_eth.npy는 제외
        npy_files = [
            f for f in os.listdir(directory) 
            if f.endswith('.npy') 
            and f not in [j+"_targets.npy", j+".npy"] and f.startswith(j)
        ]
        # 각 예측 파일에 대해 ADE, FDE 계산
        for file_name in npy_files:
            pred_path = os.path.join(directory, file_name)

            predictions = np.load(pred_path)  # shape: (876, 12, 44, 2)
            ade, fde = compute_ADE_FDE(predictions, targets)
            print(f"{file_name} => ADE: {ade:.4f}, FDE: {fde:.4f}")
