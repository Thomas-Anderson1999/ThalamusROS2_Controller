"""
ThalamusRecon(recon_modeling.py)의 new_segmentation()에서 floor mask를 뽑아내는
부분만 떼어와 camera_ws 쪽으로 이식한 모듈. (그 외 wall/object 세그멘테이션, SAM 등은
아직 포함하지 않음 - 우선 floor mask만.)

파이프라인: SLIC 슈퍼픽셀 -> LAB/깊이 전처리 -> depth 기반 인접관계 필터링
           -> 그래프 기반 러프 세그멘테이션(raw_segmentation) -> RANSAC 평면 추정
           -> floor 마스크 생성 -> floor 노이즈 정리(주변 세그먼트 편입)

원본 출처:
    ~/ThalamusWorkplace/ThalamusRecon/recon_modeling.py (new_segmentation)
    ~/ThalamusWorkplace/ThalamusRecon/daset_skelcontour/segmentation_imgprc.py
        (preprocess_for_lab, preprocess_for_depth, raw_segmentation, super_pixel_process)

주의: scikit-image(skimage)가 필요합니다. camera_ws가 쓰는 venv에 아직 없다면
      `pip install scikit-image` 로 설치해야 동작합니다.
"""
import time
import numpy as np
import cv2
import skimage
from skimage.segmentation import slic

from camera_pub.recon_util import pixel_to_vec


# ── SLIC + LAB/깊이 전처리 (원본 daset_skelcontour/segmentation_imgprc.py에서 그대로 포팅) ──

def preprocess_for_lab(labels, preprocessed_img):
    """슈퍼픽셀(labels)별로 LAB 평균/표준편차, 중심 좌표, 인접관계(label_adj)를 계산한다.

    (원본은 픽셀 하나하나를 파이썬 for y/for x 이중 루프로 도는 구현이라 300x300 라벨
    기준 호출 1번에 ~0.8초씩 걸렸다 - numpy 벡터화로 다시 짠 버전. 입출력/결과는 원본과
    동일하되, label_cnt==0 라벨의 avg/std는 원본처럼 0.0으로 남는다.)"""
    label_max = int(np.max(labels))
    h, w = labels.shape
    n = label_max + 1

    flat_labels = labels.ravel()
    L = preprocessed_img[:, :, 0].ravel().astype(np.float64)
    A = preprocessed_img[:, :, 1].ravel().astype(np.float64)
    B = preprocessed_img[:, :, 2].ravel().astype(np.float64)

    label_cnt = np.bincount(flat_labels, minlength=n).astype(np.int32)

    def _avg_std(vals):
        s   = np.bincount(flat_labels, weights=vals,       minlength=n)
        sq  = np.bincount(flat_labels, weights=vals * vals, minlength=n)
        avg = np.zeros(n, np.float64)
        std = np.zeros(n, np.float64)
        valid = label_cnt > 0
        avg[valid] = s[valid] / label_cnt[valid]
        var = sq[valid] / label_cnt[valid] - avg[valid] ** 2
        std[valid] = np.sqrt(np.maximum(var, 0.0))
        return avg, std

    label_L_avg, label_L_std = _avg_std(L)
    label_A_avg, label_A_std = _avg_std(A)
    label_B_avg, label_B_std = _avg_std(B)

    # 중심 좌표(각 라벨에 속하는 픽셀들의 x/y 평균)
    ys, xs = np.divmod(np.arange(h * w), w)  # flat index -> (y, x), row-major와 일치
    sum_x = np.bincount(flat_labels, weights=xs.astype(np.float64), minlength=n)
    sum_y = np.bincount(flat_labels, weights=ys.astype(np.float64), minlength=n)
    label_ctr = np.zeros((2, n), np.float64)
    valid = label_cnt > 0
    label_ctr[0][valid] = sum_x[valid] / label_cnt[valid]
    label_ctr[1][valid] = sum_y[valid] / label_cnt[valid]

    # 인접관계(label_adj) + sp_edge: 오른쪽/아래쪽 이웃과 라벨이 다른 경계 픽셀만 본다
    # (전체 h*w 픽셀이 아니라 경계 픽셀만 순회하므로 훨씬 적음)
    sp_edge = -np.ones((h, w, 2), dtype=np.int32)
    label_adj = {i: set() for i in range(n)}

    if w > 1:
        diff_r = labels[:, :-1] != labels[:, 1:]
        a_r = labels[:, :-1][diff_r]
        b_r = labels[:, 1:][diff_r]
        sp_edge[:, :-1, 0][diff_r] = a_r
        sp_edge[:, :-1, 1][diff_r] = b_r
    else:
        a_r = np.empty(0, dtype=labels.dtype)
        b_r = np.empty(0, dtype=labels.dtype)

    if h > 1:
        diff_d = labels[:-1, :] != labels[1:, :]
        a_d = labels[:-1, :][diff_d]
        b_d = labels[1:, :][diff_d]
        # 원본은 오른쪽 체크 다음에 아래쪽을 체크해서, 둘 다 다르면 아래쪽 관계로 덮어썼다 -> 순서 유지
        sp_edge[:-1, :, 0][diff_d] = a_d
        sp_edge[:-1, :, 1][diff_d] = b_d
    else:
        a_d = np.empty(0, dtype=labels.dtype)
        b_d = np.empty(0, dtype=labels.dtype)

    all_a = np.concatenate([a_r, a_d])
    all_b = np.concatenate([b_r, b_d])
    if all_a.size:
        pairs = np.unique(np.stack([all_a, all_b], axis=1), axis=0)
        for u, v in pairs.tolist():
            label_adj[u].add(v)
            label_adj[v].add(u)

    # first_pos: 그 라벨이 래스터 스캔(위->아래, 왼->오) 순서로 처음 나오는 픽셀 위치
    first_flat_idx = np.full(n, h * w, dtype=np.int64)  # h*w = "아직 없음" sentinel
    np.minimum.at(first_flat_idx, flat_labels, np.arange(h * w))
    first_pos = [None if (label_cnt[i] == 0) else
                 (int(first_flat_idx[i] % w), int(first_flat_idx[i] // w))
                 for i in range(n)]

    return (label_max, label_cnt, label_ctr,
            label_L_avg, label_A_avg, label_B_avg,
            label_L_std, label_A_std, label_B_std,
            sp_edge, label_adj, first_pos)


def preprocess_for_depth(labels, depth_img, min_val=100, max_val=None):
    """슈퍼픽셀별 depth 평균/표준편차 (min_val~max_val 범위 밖 값은 제외)."""
    label_max = np.max(labels)
    depth = depth_img.astype(np.float64)

    flat_labels = labels.ravel()
    flat_depth = depth.ravel()
    valid_mask = flat_depth >= min_val
    if max_val is not None:
        valid_mask &= flat_depth <= max_val
    flat_labels = flat_labels[valid_mask]
    flat_depth = flat_depth[valid_mask]

    label_cnt = np.bincount(flat_labels, minlength=label_max + 1)
    depth_sum = np.bincount(flat_labels, weights=flat_depth, minlength=label_max + 1)
    depth_sqsum = np.bincount(flat_labels, weights=flat_depth ** 2, minlength=label_max + 1)

    depth_avg = np.zeros(label_max + 1, np.float64)
    depth_std = np.zeros(label_max + 1, np.float64)

    valid = label_cnt > 0
    depth_avg[valid] = depth_sum[valid] / label_cnt[valid]
    var = depth_sqsum[valid] / label_cnt[valid] - depth_avg[valid] ** 2
    depth_std[valid] = np.sqrt(np.maximum(var, 0.0))

    return depth_avg, depth_std


def raw_segmentation(labels, label_max, label_cnt, label_L_avg, label_A_avg, label_B_avg, label_adj,
                      L_th=1.0, A_th=1.0, B_th=1.0, k=4000.0):
    """그래프 기반(Felzenszwalb 스타일) union-find 병합으로 과분할 슈퍼픽셀을 러프하게 합친다."""
    num_sp = label_max + 1
    parent = np.arange(num_sp, dtype=np.int32)
    size = label_cnt.copy().astype(np.int32)
    interval_weight = np.zeros(num_sp, dtype=np.float64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, w):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        interval_weight[ra] = max(interval_weight[ra], interval_weight[rb], w)

    edges = []
    for u in range(num_sp):
        for v in label_adj.get(u, []):
            if v <= u:
                continue
            dL = label_L_avg[u] - label_L_avg[v]
            dA = label_A_avg[u] - label_A_avg[v]
            dB = label_B_avg[u] - label_B_avg[v]
            d = np.sqrt(L_th * dL * dL + A_th * dA * dA + B_th * dB * dB)
            edges.append((d, u, v))
    edges.sort(key=lambda t: t[0])

    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if w <= min(interval_weight[ru] + k / max(1, size[ru]),
                    interval_weight[rv] + k / max(1, size[rv])):
            union(ru, rv, w)

    root_map = np.array([find(i) for i in range(num_sp)])
    merged_labels = root_map[labels]
    unique_roots, new_ids = np.unique(merged_labels, return_inverse=True)
    merged_labels = new_ids.reshape(merged_labels.shape) + 2  # 0, 1은 사용하지 않고 2부터 시작

    return merged_labels


def super_pixel_process_img(img_bgr, segment_num, compactness, proc_size, _sigma=1):
    """원본 super_pixel_process()를 파일이 아니라 메모리 상의 BGR 이미지를 받도록 수정한 버전."""
    resized_img = cv2.resize(img_bgr, proc_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) / 255.0
    lab_img = skimage.color.rgb2lab(rgb)
    labels = slic(lab_img, n_segments=segment_num, compactness=compactness, start_label=1, sigma=_sigma)
    return resized_img, labels, lab_img


# ── floor mask 추출 (원본 recon_modeling.py new_segmentation의 +floor estimation ~ -floor clear adg 부분) ──

def predict_floor_ours(rgb_bgr, depth_mm, floor_pos, floor_opt,
                        slic_compactness=10, slic_segment_num=300,
                        effective_depth=(300.0, 6000.0),
                        proc_size=(300, 300), target_shape=(720, 1280),
                        depth_th=2.0, f_th=6.0):
    """
    rgb_bgr:  원본 해상도 BGR 이미지
    depth_mm: 원본 해상도 depth map (mm 단위)
    floor_pos: (x, y) - proc_size(기본 300x300) 좌표계 기준, 바닥이라고 확정지은 위치
    floor_opt: (floor_depth_th, floor_label_th)
        floor_depth_th(mm): 평면과의 거리가 이 안이면 바닥이라 간주
        floor_label_th:     러프 세그먼트 안에서 이 비율 이상이 바닥이면 세그먼트 전체를 바닥으로 간주
    반환: (floor_label_mask, floor_vis, raugh_merged_labels, labels, depth_resized)
        floor_label_mask: labels와 같은 크기의 bool mask
        floor_vis:        floor_label_mask를 0/255로 시각화한 uint8 이미지
        raugh_merged_labels: 러프 세그먼트 라벨(바닥은 1로 통일됨)
        labels:           SLIC 슈퍼픽셀 라벨
        depth_resized:    labels와 같은 크기로 리사이즈된 depth(mm) - birdview 등 후속 계산에 재사용
    """
    floor_pixel_x, floor_pixel_y = floor_pos
    floor_depth_th, floor_label_th = floor_opt
    eff_depth_min, eff_depth_max = effective_depth

    _t0 = time.time()

    # 1) SLIC 슈퍼픽셀
    resized_img, labels, lab_img = super_pixel_process_img(rgb_bgr, slic_segment_num, slic_compactness, proc_size)
    label_h, label_w = labels.shape
    depth_resized = cv2.resize(depth_mm, (label_w, label_h), interpolation=cv2.INTER_NEAREST)
    _t1 = time.time()
    print(f"[predict_floor_ours][timing] 1) SLIC: {_t1 - _t0:.3f}s")

    (slic_label_max, slic_label_cnt, slic_label_ctr,
     slic_label_L_avg, slic_label_A_avg, slic_label_B_avg,
     slic_label_L_std, slic_label_A_std, slic_label_B_std,
     slic_sp_edge, slic_label_adj, _) = preprocess_for_lab(labels, lab_img.astype(np.float32))

    depth_avg, depth_std = preprocess_for_depth(labels, depth_resized, min_val=eff_depth_min, max_val=eff_depth_max)
    _t2 = time.time()
    print(f"[predict_floor_ours][timing] 2) preprocess_for_lab(1st)+preprocess_for_depth: {_t2 - _t1:.3f}s")

    # 2) depth 기반 인접관계 필터링: 급격히 depth가 튀는 이웃관계는 끊는다.
    for i1 in list(slic_label_adj.keys()):
        for i2 in list(slic_label_adj[i1]):
            if not (eff_depth_min <= depth_avg[i1] <= eff_depth_max and eff_depth_min <= depth_avg[i2] <= eff_depth_max):
                continue  # 둘 중 하나라도 effective_depth 범위 밖이면 depth 값을 신뢰할 수 없으므로 adj를 건드리지 않는다.
            if depth_th * (depth_std[i1] + depth_std[i2]) < abs(depth_avg[i1] - depth_avg[i2]):
                slic_label_adj[i1].discard(i2)
                slic_label_adj[i2].discard(i1)
            if f_th < (depth_std[i1] ** 2) / (depth_std[i2] ** 2):
                slic_label_adj[i1].discard(i2)
                slic_label_adj[i2].discard(i1)
    _t3 = time.time()
    print(f"[predict_floor_ours][timing] 3) depth 인접관계 필터링: {_t3 - _t2:.3f}s")

    # 3) 러프 세그멘테이션(과분할된 슈퍼픽셀을 LAB 기준으로 러프하게 병합)
    raugh_merged_labels = raw_segmentation(labels, slic_label_max, slic_label_cnt,
                                           slic_label_L_avg, slic_label_A_avg, slic_label_B_avg, slic_label_adj,
                                           L_th=1.0, A_th=4.0, B_th=4.0, k=700.0)
    _t4 = time.time()
    print(f"[predict_floor_ours][timing] 4) raw_segmentation: {_t4 - _t3:.3f}s")

    # 4) floor_pos 위치가 속한 러프 세그먼트의 3D 포인트로 평면을 RANSAC 추정
    floor_idx = raugh_merged_labels[floor_pixel_y][floor_pixel_x]
    floor_segment_idx = 1

    floor_ys, floor_xs = np.where(raugh_merged_labels == floor_idx)
    _t_pv0 = time.time()
    floor_pts = np.array([pixel_to_vec(px, py, target_shape, depth_resized) for px, py in zip(floor_xs, floor_ys)])
    _t_pv1 = time.time()
    print(f"[predict_floor_ours][timing] 5) floor_pts용 pixel_to_vec 루프: {_t_pv1 - _t_pv0:.3f}s "
          f"({len(floor_xs)}개 픽셀, {(_t_pv1 - _t_pv0) / max(1, len(floor_xs)) * 1000:.4f}ms/픽셀)")
    floor_pts = floor_pts[floor_pts[:, 2] > 0]  # depth 0인 무효 포인트 제거

    normal, d = None, 0.0
    if len(floor_pts) >= 3:
        ransac_iter = 200
        rng = np.random.default_rng()
        best_inlier_cnt = 0

        for _ in range(ransac_iter):
            idx = rng.choice(len(floor_pts), 3, replace=False)
            p0, p1, p2 = floor_pts[idx]

            v1 = p1 - p0
            v2 = p2 - p0
            cand_normal = np.cross(v1, v2)
            n_len = np.linalg.norm(cand_normal)
            if n_len < 1e-6:  # 세 점이 일직선상에 있어 평면을 못 만드는 경우 skip
                continue
            cand_normal = cand_normal / n_len
            cand_d = np.dot(cand_normal, p0)

            dists = np.abs(floor_pts @ cand_normal - cand_d)
            inlier_cnt = int(np.sum(dists < floor_depth_th))

            if inlier_cnt > best_inlier_cnt:
                best_inlier_cnt = inlier_cnt
                normal = cand_normal
                d = cand_d
        print(f"[predict_floor_ours][RANSAC] inliers={best_inlier_cnt}/{len(floor_pts)}")
    _t5 = time.time()
    print(f"[predict_floor_ours][timing] 6) RANSAC 평면 추정: {_t5 - _t_pv1:.3f}s")

    print(f"[predict_floor_ours] plane normal={normal}, d={d}, pts={len(floor_pts)}")

    # 5) 슈퍼픽셀 중심이 평면 위(floor_depth_th 이내)에 있으면 floor_label_mask에 편입
    floor_label_mask = np.zeros_like(labels, dtype=bool)
    _t_pv2 = time.time()
    _sp_checked = 0
    if normal is not None:
        for sp_idx in range(slic_label_max + 1):
            if slic_label_cnt[sp_idx] == 0:
                continue
            cx, cy = int(slic_label_ctr[0][sp_idx]), int(slic_label_ctr[1][sp_idx])
            p = pixel_to_vec(cx, cy, target_shape, depth_resized)
            _sp_checked += 1
            if p[2] <= 0:
                continue
            if abs(np.dot(normal, p) - d) < floor_depth_th:
                floor_label_mask |= (labels == sp_idx)
    _t6 = time.time()
    print(f"[predict_floor_ours][timing] 7) 슈퍼픽셀 중심 pixel_to_vec 루프: {_t6 - _t_pv2:.3f}s ({_sp_checked}개 슈퍼픽셀)")

    # 5.1) floor_label_mask에 드문드문 뚫린 구멍: 같은 러프 세그먼트 안에서 floor 비율이 floor_label_th
    #      이상이면 그 세그먼트 전체를 floor로 편입한다.
    raugh_max = int(np.max(raugh_merged_labels))
    total_cnt = np.bincount(raugh_merged_labels.ravel(), minlength=raugh_max + 1)
    floor_cnt = np.bincount(raugh_merged_labels[floor_label_mask], minlength=raugh_max + 1)

    ratio = np.divide(floor_cnt, total_cnt, out=np.zeros_like(floor_cnt, dtype=np.float64), where=total_cnt > 0)
    floor_ids = np.where(ratio >= floor_label_th)[0]
    floor_label_mask |= np.isin(raugh_merged_labels, floor_ids)

    # 5.2) floor_label_mask에 해당하는 raugh_merged_labels는 floor_segment_idx(1)로 통일
    raugh_merged_labels[floor_label_mask] = floor_segment_idx
    _t7 = time.time()
    print(f"[predict_floor_ours][timing] 8) floor 구멍 메우기(bincount): {_t7 - _t6:.3f}s")

    # 6) floor 노이즈 정리: label_adj가 오직 floor뿐인(=floor에만 붙어있는) 세그먼트를
    #    depth/LAB 통계가 이웃 floor 세그먼트와 비슷하면 floor로 편입한다.
    (label_max, label_cnt, label_ctr,
     label_L_avg, label_A_avg, label_B_avg,
     label_L_std, label_A_std, label_B_std,
     sp_edge, label_adj, _) = preprocess_for_lab(raugh_merged_labels, lab_img.astype(np.float32))
    _t8 = time.time()
    print(f"[predict_floor_ours][timing] 9) preprocess_for_lab(2nd): {_t8 - _t7:.3f}s")

    noise_ids = []
    for lbl, adj in label_adj.items():
        if lbl == floor_segment_idx:  # floor인 경우 skip
            continue
        if floor_segment_idx not in adj:  # floor와 인접하지 않으면 skip
            continue

        depth_noise_th = 1.5
        l_noise_th = 1.5
        a_noise_th = 1.5
        b_noise_th = 1.5

        x, y = (int(label_ctr[0][lbl]), int(label_ctr[1][lbl]))
        slic_lbl = labels[y][x]
        neighbor_ids = [nid for nid in slic_label_adj.get(slic_lbl, set())
                        if floor_label_mask[int(slic_label_ctr[1][nid]), int(slic_label_ctr[0][nid])]]
        if not neighbor_ids:  # 비교할 floor 이웃이 없으면 통계적으로 검증 불가하므로 병합하지 않는다.
            continue

        neighbor_depth_avg = np.mean(depth_avg[neighbor_ids])
        neighbor_depth_std = np.mean(depth_std[neighbor_ids])
        if abs(depth_avg[slic_lbl] - neighbor_depth_avg) > depth_noise_th * neighbor_depth_std:
            continue

        neighbor_L_avg = np.mean(slic_label_L_avg[neighbor_ids])
        neighbor_A_avg = np.mean(slic_label_A_avg[neighbor_ids])
        neighbor_B_avg = np.mean(slic_label_B_avg[neighbor_ids])
        neighbor_L_std = np.mean(slic_label_L_std[neighbor_ids])
        neighbor_A_std = np.mean(slic_label_A_std[neighbor_ids])
        neighbor_B_std = np.mean(slic_label_B_std[neighbor_ids])

        if abs(slic_label_L_avg[slic_lbl] - neighbor_L_avg) > l_noise_th * neighbor_L_std:
            continue
        if abs(slic_label_A_avg[slic_lbl] - neighbor_A_avg) > a_noise_th * neighbor_A_std:
            continue
        if abs(slic_label_B_avg[slic_lbl] - neighbor_B_avg) > b_noise_th * neighbor_B_std:
            continue

        noise_ids.append(lbl)

    if noise_ids:
        noise_pos = [(lbl, (int(label_ctr[0][lbl]), int(label_ctr[1][lbl]))) for lbl in noise_ids]
        print(f"[predict_floor_ours][floor noise clearing] merged into floor: {noise_pos}")
        noise_mask = np.isin(raugh_merged_labels, noise_ids)
        floor_label_mask |= noise_mask
        raugh_merged_labels[noise_mask] = floor_segment_idx
    _t9 = time.time()
    print(f"[predict_floor_ours][timing] 10) floor 노이즈 정리 루프: {_t9 - _t8:.3f}s "
          f"({len(label_adj)}개 세그먼트 검사)")

    # 7) floor 시각화
    floor_vis = np.zeros((label_h, label_w), dtype=np.uint8)
    floor_vis[floor_label_mask] = 255

    print(f"[predict_floor_ours][timing] === 총합: {_t9 - _t0:.3f}s ===")
    return floor_label_mask, floor_vis, raugh_merged_labels, labels, depth_resized
