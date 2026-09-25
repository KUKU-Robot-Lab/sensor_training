"""각도-프리 변형 복원기 학습 (Phase 2·3) — due 데이터만으로 self-supervised.

3중 손실(설계 확정, Notion 토글 기록):
  L_suppress = ||restore(변형 무접촉)||²              → 유령 억제(무료 라벨)
  L_identity = ||restore(baseline) − baseline||²      → 변형 없으면 그대로(과잉보정 방지)
  L_contact  = ||restore(변형+접촉) − 접촉||²         → ★접촉 보존(붕괴 방지 핵심)

L_contact 의 접촉은 **기존 flat xy 취득 데이터**의 pct 를 재사용(§5.4 가법 가정).
실측 검증(2026-08-08, v6_new): L_contact 없으면 억제 36%·loc 4.26mm,
있으면 **87%·1.00mm** — 이 손실이 붕괴 방지의 결정적 요인.

leave-one-session-out 으로 미학습 변형에서의 억제율·접촉보존을 평가한다.

실행(★repo 루트 `sensor_training/` 에서, `.venv/bin/python` 으로):
  .venv/bin/python -m sats.bending.train_deform_restorer \\
    --deform-root skin_ws/raw_data/deform/v7 --latent-dims 2 4 8

★센서별 자산 자동 매칭: --deform-root 경로의 vN(또는 --sensor v7)에서
  contact-trial = learning_data/sensor_raw_bin/ecomesh_vN_xy1/d5/z_2.5mm/test1
  sats-run      = sats/training/runs/ecomesh_vN_deploy_g025 (없으면 all4/g01)
을 자동 지정한다. **변형 데이터와 다른 센서의 접촉/SATS를 섞으면 안 되므로**
(캘리브레이션 불일치) 없으면 실행을 중단하고 안내한다. 필요하면 직접 지정도 가능.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .baseline_restorer import BaselineRestorer
from .config import BendingConfig
from .deform_data import MAG_BINS, MAG_NAMES, deform_windows, load_all, window_magnitude

_REPO = Path(__file__).resolve().parents[2]


def _contact_pool(trial_dir: Path, window: int, n: int, fz_min: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """flat 접촉 trial → (pct 윈도우[K,W,16], 실제 접촉 좌표[K,2] mm)."""
    from .eval_contact_preservation import contact_windows
    return contact_windows(trial_dir, window, n, fz_min)


@torch.no_grad()
def contact_only_localization(pool: np.ndarray, xy_true: np.ndarray, sats, device: str) -> dict:
    """★변형이 전혀 없을 때 SATS 가 접촉을 얼마나 잡는가 — 복원 성능의 상한선.

    복원기가 아무리 잘해도 이 값보다 좋아질 수 없다. 이 값이 이미 크면 병목은
    복원기가 아니라 센서 상태(파손 taxel)나 SATS 자체다.
    """
    from .eval_contact_preservation import _peak_xy
    C = torch.from_numpy(pool).to(device)
    from .eval_contact_preservation import _sats_map
    xy = _peak_xy(_sats_map(sats, C))
    err = np.linalg.norm(xy - xy_true[:len(xy)], axis=1)
    return {"mean_mm": float(err.mean()), "median_mm": float(np.median(err)),
            "p90_mm": float(np.percentile(err, 90)), "n": int(len(err))}


def train_deform_restorer(
    train_win: np.ndarray, base_win: np.ndarray, contact_pool: np.ndarray | None,
    cfg: BendingConfig, *, device: str = "cpu", epochs: int = 120, lr: float = 1e-3,
    lam_contact: float = 1.0, lam_identity: float = 0.5, batch: int = 512,
) -> BaselineRestorer:
    """3중 손실로 latent restorer 학습. train_win=변형 구간, base_win=앞뒤 baseline 구간."""
    X = torch.from_numpy(train_win).to(device)
    B = torch.from_numpy(base_win).to(device) if len(base_win) else None
    C = torch.from_numpy(contact_pool).to(device) if contact_pool is not None else None
    model = BaselineRestorer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            loss = nn.functional.mse_loss(model(xb), torch.zeros_like(xb))       # L_suppress
            if C is not None:                                                    # L_contact
                cb = C[torch.randint(0, len(C), (len(xb),), device=device)]
                loss = loss + lam_contact * nn.functional.mse_loss(model(xb + cb), cb)
            if B is not None and lam_identity > 0:                                # L_identity
                bb = B[torch.randint(0, len(B), (min(len(xb), len(B)),), device=device)]
                loss = loss + lam_identity * nn.functional.mse_loss(model(bb), bb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def evaluate(model: BaselineRestorer, val_win: np.ndarray, contact_pool: np.ndarray,
             sats, device: str) -> dict:
    """홀드아웃 변형에서 억제율 + 준-합성 접촉 보존."""
    from .eval_contact_preservation import _peak_xy, _sats_map
    V = torch.from_numpy(val_win).to(device)
    n = min(len(V), len(contact_pool))
    idx = np.linspace(0, len(V) - 1, n).astype(int)
    vb = V[idx]
    C = torch.from_numpy(contact_pool[:n]).to(device)
    flat = _sats_map(sats, torch.zeros(1, vb.shape[1], vb.shape[2], device=device))[0]
    # 1) 유령 억제: 무접촉 변형 → SATS 맵이 flat 에 얼마나 가까운가
    m_unc = _sats_map(sats, vb)
    m_cor = _sats_map(sats, model(vb))
    du = (m_unc - flat).abs().mean(dim=(1, 2)).mean().item()
    dc = (m_cor - flat).abs().mean(dim=(1, 2)).mean().item()
    # 2) 접촉 보존: 변형+접촉 복원 후 접촉만 맵과 비교
    ref = _sats_map(sats, C)
    unc = _sats_map(sats, vb + C)
    cor = _sats_map(sats, model(vb + C))
    eu = (unc - ref).abs().mean(dim=(1, 2)).mean().item()
    ec = (cor - ref).abs().mean(dim=(1, 2)).mean().item()
    xy_ref, xy_unc, xy_cor = _peak_xy(ref), _peak_xy(unc), _peak_xy(cor)
    return {
        "suppression": 1.0 - dc / du if du > 0 else 0.0,
        "ghost_dev_uncorrected": du, "ghost_dev_corrected": dc,
        "contact_recovery": 1.0 - ec / eu if eu > 0 else 0.0,
        "loc_uncorrected_mm": float(np.linalg.norm(xy_unc - xy_ref, axis=1).mean()),
        "loc_corrected_mm": float(np.linalg.norm(xy_cor - xy_ref, axis=1).mean()),
        "n": int(n),
    }


def evaluate_by_magnitude(model: BaselineRestorer, val_win: np.ndarray, pool: np.ndarray,
                          sats, device: str, bins=MAG_BINS) -> list[dict]:
    """변형 **크기 구간별** 성능 — 평균 하나로는 답할 수 없는 질문에 답한다.

    "복원이 안 되는가"와 "특정 크기 이상에서만 안 되는가"는 완전히 다른 결론으로 이어진다.
    전자면 방법을 바꿔야 하고, 후자면 **데모에서 쓸 변형 범위를 정하면** 되기 때문이다.
    구간 기준은 **채널 평균 |pct|**(센서 전체가 얼마나 휘었는가). 최댓값을 쓰면 파손
    채널 하나가 통계를 지배하므로 쓰지 않는다 — deform_data.MAG_BINS 주석 참조.
    """
    mag = window_magnitude(val_win)
    out = []
    for lo, hi in bins:
        sel = (mag >= lo) & (mag < hi)
        if sel.sum() < 8:                       # 표본이 너무 적으면 수치가 의미 없다
            continue
        r = evaluate(model, val_win[sel], pool, sats, device)
        r.update({"mag_lo": lo, "mag_hi": None if np.isinf(hi) else hi,
                  "n_windows": int(sel.sum())})
        out.append(r)
    return out


_DROPOUT_PCT = 90.0          # |pct| 가 이보다 크면 raw≈0 = 전송 드롭아웃


def drop_dropout_windows(win: np.ndarray, *, quiet: bool = False) -> np.ndarray:
    """드롭아웃(raw=0 → pct −100%) 프레임이 섞인 윈도우 제거.

    ★파손이 진행 중인 채널은 간헐적으로 raw=0 을 낸다. 그 프레임의 pct 는 −100% 로,
    실제 변형(수 %)의 20배가 넘는다. L_suppress 는 "이걸 0 으로 만들라"고 가르치므로
    모델이 변형이 아니라 드롭아웃을 지우는 데 용량을 쓴다. 프레임 수는 적어도
    윈도우 단위로는 크게 번진다(실측 test3: 183프레임 → 윈도우 약 6.6%).

    채널을 통째로 마스킹하지 않는 이유: 그 채널이 변형 신호의 최대 기여자인 경우가 있어
    (실측 S14: 4세션 모두 채널 p99 1위) 빼면 관측력이 크게 준다.
    """
    if len(win) == 0:
        return win
    keep = ~(np.abs(win) > _DROPOUT_PCT).any(axis=(1, 2))
    n_drop = int((~keep).sum())
    if n_drop and not quiet:
        print(f"    드롭아웃 윈도우 {n_drop}개 제외({n_drop / len(win) * 100:.1f}%)")
    return win[keep]


def _drop_contact_contaminated(win: np.ndarray, max_mag: float, *, quiet: bool = False
                               ) -> np.ndarray:
    """크기가 max_mag 를 넘는 윈도우 제거 — 손가락 압박(접촉) 오염 방지.

    ★현장 관찰: 손으로 변형만 시키면 채널 평균 |pct| 가 10% 를 넘지 않는다. 넘는 경우는
    활성면을 손가락으로 **누른** 것이고, 그건 변형이 아니라 접촉이다. 이런 윈도우가
    L_suppress(=출력 0) 에 들어가면 모델은 "손가락 접촉은 지워라"를 학습하고,
    데모에서 관람객의 실제 터치까지 지워버린다. L_contact 로도 못 막는다 —
    상반된 두 라벨을 같은 신호에 주는 셈이기 때문이다.
    """
    if max_mag <= 0 or len(win) == 0:
        return win
    keep = window_magnitude(win) <= max_mag
    n_drop = int((~keep).sum())
    if n_drop and not quiet:
        print(f"    접촉 오염 의심 윈도우 {n_drop}개 제외(크기 > {max_mag:.0f}%, "
              f"{n_drop / len(win) * 100:.1f}%)")
    return win[keep]


def _rel(path: Path) -> str:
    """저장소 기준 상대경로로 표기(저장소 밖이면 절대경로 그대로)."""
    path = Path(path)
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def _contact_search_roots(sensor: str) -> list[Path]:
    """접촉 trial 이 있을 수 있는 두 레이아웃 — PC 마다 어느 쪽만 있을 수 있다."""
    return [_REPO / f"learning_data/sensor_raw_bin/ecomesh_{sensor}_xy1",   # 정리된 학습 데이터
            _REPO / f"skin_ws/raw_data/sats/ecomesh/{sensor}"]              # 원시 취득(병합 후)


def _find_contact_trial(sensor: str) -> Path | None:
    """센서의 xy 접촉 trial 을 **실제로 존재하는 것 중에서** 고른다.

    d5/z_2.5mm/test1 처럼 조합을 하드코딩하면 PC 마다 취득 조합이 달라 실패한다.
    contact_windows 는 `*_merged.bin` 과 `*_baseline.json` 을 함께 요구하므로 둘 다 있는
    폴더만 후보로 본다(원시 bin 만 있는 미병합 폴더를 잘못 고르지 않도록).
    d5(작은 인덴터)를 우선 — 접촉이 국소적이라 L_contact 의 보존 판정이 더 엄격하다.
    """
    trials: set[Path] = set()
    for root in _contact_search_roots(sensor):
        if not root.exists():
            continue
        trials |= {b.parent for b in root.rglob("*_merged.bin")
                   if any(b.parent.glob("*_baseline.json"))}
    if not trials:
        return None
    return min(trials, key=lambda p: (0 if "/d5" in f"{p}/" else 1, str(p)))


def _resolve_dead_channels(spec: str, deform_root) -> tuple[int, ...]:
    """'auto'|'none'|'11,15'(1-based) → 마스킹할 0-based 채널.

    auto 는 세션들을 진단해 **영구 고장(dead/faulty)의 합집합**만 마스킹한다. 회복한
    일시적 드롭아웃(glitchy)은 멀쩡한 taxel 이므로 제외한다.
    """
    spec = (spec or "auto").strip().lower()
    if spec in ("none", ""):
        return ()
    if spec != "auto":
        try:
            ch = tuple(sorted({int(x) - 1 for x in spec.replace(" ", "").split(",") if x}))
        except ValueError:
            raise SystemExit(f"--dead-channels 형식 오류: {spec!r} (예: '11,15')")
        if any(c < 0 or c > 15 for c in ch):
            raise SystemExit(f"--dead-channels 는 1~16 범위여야 함: {spec!r}")
        print(f"[채널] 수동 마스킹: {','.join(f'S{c + 1:02d}' for c in ch)}")
        return ch

    from sats.tools.channel_health import analyze_channels, bad_indices
    from .deform_data import discover_sessions, read_due_raw
    found: set[int] = set()
    for d in discover_sessions(deform_root):
        t, raw = read_due_raw(d)
        span = float(t[-1] - t[0])
        try:
            rep = analyze_channels(raw, fs=(len(t) / span if span > 1e-6 else 200.0))
        except ValueError as e:
            print(f"  [{d.name}] 진단 건너뜀 — {e}")
            continue
        found |= set(bad_indices(rep))
    if found:
        print(f"[채널] ★자동 마스킹: {','.join(f'S{c + 1:02d}' for c in sorted(found))} "
              f"(영구 고장 — 학습·추론 양쪽에 동일 적용할 것)")
    else:
        print("[채널] 전 세션 16채널 정상 — 마스킹 없음")
    return tuple(sorted(found))


def main() -> None:
    p = argparse.ArgumentParser(description="각도-프리 변형 복원기 학습(due 데이터만)")
    p.add_argument("--deform-root", required=True, help="변형 세션 루트(due bin 폴더들)")
    p.add_argument("--sensor", default=None,
                   help="★센서 버전(예: v7) — contact-trial·sats-run 을 그 센서 것으로 자동 지정. "
                        "미지정 시 --deform-root 경로에서 vN 추출 시도")
    p.add_argument("--contact-trial", type=Path, default=None,
                   help="L_contact·평가용 flat 접촉 trial(미지정=--sensor 로 자동)")
    p.add_argument("--sats-run", type=Path, default=None,
                   help="동결 SATS run(미지정=--sensor 로 자동)")
    p.add_argument("--latent-dims", type=int, nargs="+", default=[2, 4, 8],
                   help="잠재 차원 스윕(억제율 vs 접촉보존 트레이드오프)")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--n-contact", type=int, default=300)
    p.add_argument("--fz-min", type=float, default=0.5)
    p.add_argument("--no-contact-loss", action="store_true", help="L_contact 제거(ablation)")
    p.add_argument("--max-magnitude", type=float, default=10.0,
                   help="★이 크기(채널 평균 |pct| %%)를 넘는 변형 윈도우는 학습에서 제외. "
                        "그 영역은 손가락 압박(=접촉)이 섞인 것이라, L_suppress 에 들어가면 "
                        "모델이 '접촉은 지워야 할 것'으로 배운다. 0 이면 제외 안 함")
    p.add_argument("--dead-channels", default="auto",
                   help="파손 taxel(1-based, 예 '11,15'). 'auto'=채널 건강도 진단으로 "
                        "자동 검출(기본) · 'none'=마스킹 안 함")
    p.add_argument("--out", type=Path, default=None,
                   help="저장 폴더(기본: runs/deform_restorer_<센서> — 대시보드 restore "
                        "드롭다운에 센서별로 나열된다)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # ── 센서 버전 해석: --sensor > deform-root 경로의 vN > 오류 ──────────────
    sensor = args.sensor
    if sensor is None:
        import re
        m = re.search(r"\bv(\d+)\b", str(args.deform_root))
        sensor = f"v{m.group(1)}" if m else None
    if sensor is None and (args.contact_trial is None or args.sats_run is None):
        raise SystemExit("센서 버전을 알 수 없습니다 — --sensor v7 처럼 지정하거나 "
                         "--contact-trial/--sats-run 을 직접 주세요")
    if args.contact_trial is None:
        found = _find_contact_trial(sensor)
        if found is None:
            roots = _contact_search_roots(sensor)
            lines = [f"[{sensor}] xy 접촉 trial(merged.bin + baseline.json)을 찾지 못했습니다."]
            raw_root = roots[1]
            for r in roots:
                lines.append(f"  탐색: {r}  →  {'있음' if r.exists() else '없음'}")
            if raw_root.exists():
                lines += [
                    "  ★원시 데이터는 있으나 **병합 전** 입니다. 병합부터 하세요:",
                    f"    .venv/bin/python -m sats.preprocessing.bin_merge \\",
                    f"      --raw-root {raw_root.relative_to(_REPO)}/xy_1mm/d5",
                    "    (d10 도 쓰려면 마지막을 d10 으로 바꿔 한 번 더)",
                ]
            else:
                lines.append("  → --contact-trial 로 직접 지정하거나 해당 센서 xy 데이터를 가져오세요.")
            lines.append("  ★다른 센서의 접촉 trial 로 대체하면 캘리브레이션이 어긋나므로 금지.")
            raise SystemExit("\n".join(lines))
        args.contact_trial = found
        print(f"[센서 {sensor}] 접촉 trial 자동 선택: {_rel(found)}")
    if args.sats_run is None:
        cand = [_REPO / f"sats/training/runs/{pre}{sensor}_deploy_{t}"
                for t in ("g025", "all4", "g01") for pre in ("ecomesh_", "")]
        args.sats_run = next((c for c in cand if (c / "best_model.pt").exists()), None)
        if args.sats_run is None:
            raise SystemExit(f"[{sensor}] SATS run 없음. 확인한 경로:\n  "
                             + "\n  ".join(str(c) for c in cand)
                             + "\n  → --sats-run 으로 직접 지정")
    for label, path in (("contact-trial", args.contact_trial), ("sats-run", args.sats_run)):
        if not Path(path).exists():
            raise SystemExit(f"[{sensor}] {label} 없음: {path}\n"
                             f"  → 해당 센서의 xy 취득/학습이 되어 있는지 확인하거나 직접 지정")
    print(f"[센서 {sensor}] contact={_rel(args.contact_trial)}  sats={args.sats_run.name}")

    if args.out is None:
        args.out = _REPO / f"sats/bending/runs/deform_restorer_{sensor or 'x'}"
    cfg0 = BendingConfig()
    W = cfg0.window_size
    dead = _resolve_dead_channels(args.dead_channels, args.deform_root)
    print(f"[1/3] 변형 세션 로드: {args.deform_root}")
    sessions = load_all(args.deform_root, dead_channels=dead)
    if len(sessions) < 2:
        raise SystemExit("leave-one-session-out 평가에 최소 2세션 필요")

    print(f"[2/3] 접촉 pool 로드: {args.contact_trial.name}")
    pool, pool_xy = _contact_pool(args.contact_trial, W, args.n_contact, args.fz_min)
    if dead:
        # ★접촉 pool 은 파손 전 xy 데이터라 해당 채널이 살아있다. 그대로 쓰면 변형 입력에서는
        # 항상 0 인 채널에 접촉 신호가 실려, 모델이 존재하지 않는 taxel 을 근거로 학습한다.
        pool = pool.copy()
        pool[:, :, list(dead)] = 0.0
    print(f"  접촉 윈도우 {len(pool)}개" + (f" (S{'/S'.join(f'{c + 1:02d}' for c in dead)} 마스킹)"
                                          if dead else ""))

    from .pipeline import load_frozen_sats
    sats = load_frozen_sats(args.sats_run, args.device)
    base_loc = contact_only_localization(pool, pool_xy, sats, args.device)
    print(f"  ★기준선(변형 없음) 접촉 위치오차: 평균 {base_loc['mean_mm']:.2f}mm  "
          f"중앙 {base_loc['median_mm']:.2f}mm  p90 {base_loc['p90_mm']:.2f}mm")
    print("    → 복원기는 이 값보다 좋아질 수 없다. 크면 병목은 센서/SATS 쪽.")

    print(f"[3/3] leave-one-session-out × latent_dim {args.latent_dims}")
    results: dict[str, list] = {}
    for k in args.latent_dims:
        cfg = replace(cfg0, restorer_mode="latent", latent_dim=k)
        per_session = []
        for i, held in enumerate(sessions):
            tr = [s for j, s in enumerate(sessions) if j != i]
            train_win = _drop_contact_contaminated(drop_dropout_windows(
                np.concatenate([deform_windows(s, W) for s in tr])), args.max_magnitude)
            base_win = np.concatenate([deform_windows(s, W, include_baseline=True) for s in tr])
            val_win = drop_dropout_windows(deform_windows(held, W), quiet=True)
            if len(val_win) == 0:
                continue
            model = train_deform_restorer(
                train_win, base_win, None if args.no_contact_loss else pool,
                cfg, device=args.device, epochs=args.epochs)
            r = evaluate(model, val_win, pool, sats, args.device)
            r["held_out"] = held.name
            r["by_magnitude"] = evaluate_by_magnitude(model, val_win, pool, sats, args.device)
            per_session.append(r)
            print(f"  z={k} held={held.name:12s} 억제 {r['suppression']*100:5.1f}%  "
                  f"접촉회복 {r['contact_recovery']*100:5.1f}%  "
                  f"loc {r['loc_uncorrected_mm']:.2f}→{r['loc_corrected_mm']:.2f}mm")
            for b in r["by_magnitude"]:          # ★크기별 — 데모 범위 결정의 근거
                name = next((n for n, (a, _) in zip(MAG_NAMES, MAG_BINS)
                             if a == b["mag_lo"]), "?")
                rng = f"변형 {name}%"
                print(f"      {rng:16s} n={b['n_windows']:5d}  억제 {b['suppression']*100:5.1f}%"
                      f"  loc {b['loc_uncorrected_mm']:5.2f}→{b['loc_corrected_mm']:5.2f}mm")
        if per_session:
            results[f"latent:{k}"] = per_session
            sup = np.mean([r["suppression"] for r in per_session])
            rec = np.mean([r["contact_recovery"] for r in per_session])
            loc = np.mean([r["loc_corrected_mm"] for r in per_session])
            print(f"  ★ z={k} 평균: 억제 {sup*100:.1f}%  접촉회복 {rec*100:.1f}%  loc {loc:.2f}mm")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "loso_results.json").write_text(
        json.dumps({"sessions": [s.name for s in sessions],
                    "contact_loss": not args.no_contact_loss,
                    "dead_channels_1based": [c + 1 for c in dead],
                    "contact_only_localization": base_loc,
                    "results": results}, indent=2), encoding="utf-8")
    # 최종 배포 모델: 전 세션 학습(최고 z)
    if results:
        best_k = max(results, key=lambda kk: np.mean([r["suppression"] for r in results[kk]]))
        k = int(best_k.split(":")[1])
        cfg = replace(cfg0, restorer_mode="latent", latent_dim=k)
        all_win = _drop_contact_contaminated(drop_dropout_windows(
            np.concatenate([deform_windows(s, W) for s in sessions])), args.max_magnitude)
        all_base = np.concatenate([deform_windows(s, W, include_baseline=True) for s in sessions])
        model = train_deform_restorer(all_win, all_base,
                                      None if args.no_contact_loss else pool,
                                      cfg, device=args.device, epochs=args.epochs)
        torch.save({"model": model.state_dict(), "latent_dim": k,
                    "restorer_mode": "latent",
                    # ★추론에서 동일하게 마스킹해야 한다 — 체크포인트에 함께 저장해
                    #   데모가 학습과 다른 채널 구성으로 도는 사고를 막는다.
                    "dead_channels_1based": [c + 1 for c in dead],
                    "sensor": sensor}, args.out / "best.pt")
        print(f"\n배포 모델 저장: {args.out/'best.pt'} (latent_dim={k}, 전 세션 학습)")
    print(f"결과: {args.out/'loso_results.json'}")


if __name__ == "__main__":
    main()
