import numpy as np
import os

# 디렉토리 경로 지정
dir_path = '../lobby3/test'

# .npy 파일만 추출
npy_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])

# 결과 저장
results = []

for fname in npy_files:
    file_path = os.path.join(dir_path, fname)
    
    try:
        preds = np.load(file_path)
        if preds.ndim != 4:
            raise ValueError(f"Unexpected shape: {preds.shape}")
        T, H, N, D = preds.shape
    except Exception as e:
        print(f"Skipping {fname} due to error: {e}")
        continue

    all_dists = []

    for i in range(N):  # 보행자 수
        for t in range(T - 1):  # 시간 스텝
            curr = preds[t, :, i, :]
            next = preds[t + 1, :, i, :]

            if not np.any(np.isnan(curr)) and not np.any(np.isnan(next)):
                curr_tail = curr[1:]
                next_head = next[:-1]

                dists = []
                for j in range(11):
                    dx = curr_tail[j, 0] - next_head[j, 0]
                    dy = curr_tail[j, 1] - next_head[j, 1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    dists.append(dist)

                mean_l2 = sum(dists) / 11
                all_dists.append(mean_l2)

    if all_dists:
        total_avg = sum(all_dists) / len(all_dists)
        results.append((fname, total_avg))
    else:
        results.append((fname, float('nan')))

# 결과 출력
print("\n=== Total Variation Results ===")
for fname, variation in results:
    print(f"{fname}\tTotal Variation: {variation:.6f}")
