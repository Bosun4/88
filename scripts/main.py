#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import subprocess
import traceback
import asyncio
import time
import math
from datetime import datetime, timedelta, timezone

# ============================================================
# 自动安装依赖
# ============================================================

REQUIRED_PACKAGES = [
    "aiohttp>=3.14.3",
    "Requests>=2.32.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",

]


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(float(str(os.environ.get(name, default)).strip()))
    except Exception:
        return default


def _json_safe(value):
    """Convert non-standard JSON values (NaN/Inf) to browser-parseable JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_json_atomic(path: str, payload: dict):
    """Write valid JSON atomically: temp file -> validate -> os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe_payload = _json_safe(payload)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    with open(tmp_path, "r", encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp_path, path)


def publish_prediction_outputs(data_dir: str, target_date: str, session: str, payload: dict, now_time: datetime) -> dict:
    """Publish live JSON plus immutable audit snapshots."""
    repo_dir = os.path.dirname(data_dir)
    target_path = os.path.join(data_dir, "predictions.json")
    history_path = os.path.join(data_dir, f"history_{target_date}_today_{session}.json")
    snapshot_base = os.path.join(
        data_dir,
        "snapshots",
        f"{target_date}_today_{session}_{now_time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    snapshot_path = snapshot_base
    suffix = 2
    while os.path.exists(snapshot_path):
        snapshot_path = snapshot_base[:-5] + f"_{suffix}.json"
        suffix += 1

    try:
        from .prematch_guard import enforce_publication_gate
    except ImportError:
        from prematch_guard import enforce_publication_gate
    if now_time.tzinfo is None:
        raise ValueError("Publication requires timezone-aware time")
    rows = payload.get("matches", {}).get("today", [])
    for row in rows:
        enforce_publication_gate(row, now_time)
    payload["top4"] = [row for row in payload.get("top4", [])
                       if enforce_publication_gate(row, now_time) == "eligible"]
    payload["published_at"] = now_time.isoformat()
    payload["target_date"] = target_date
    runtime = payload.setdefault("runtime", {})
    runtime["history_path"] = os.path.relpath(history_path, repo_dir).replace(os.sep, "/")
    runtime["snapshot_path"] = os.path.relpath(snapshot_path, repo_dir).replace(os.sep, "/")

    write_json_atomic(target_path, payload)
    write_json_atomic(history_path, payload)
    write_json_atomic(snapshot_path, payload)
    if rows:
        from pathlib import Path
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from forward_ledger.ledger import create_ledger_from_prediction
        ledger = os.path.join(data_dir, "forward_ledger.jsonl")
        create_ledger_from_prediction(snapshot_path, ledger, on_conflict="keep")
    return {"live": target_path, "history": history_path, "snapshot": snapshot_path}


def auto_install():
    # 默认不在运行时自动安装/升级依赖（CI 用 pip install -r requirements.txt，本地用 .venv）。
    # 运行时 pip 会污染系统环境且结果不确定；仅在显式 opt-in 时才执行。
    if not env_bool("VMAX_ALLOW_AUTO_INSTALL", False):
        return

    missing = []

    try:
        import pkg_resources

        for pkg in REQUIRED_PACKAGES:
            try:
                pkg_resources.require(pkg)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                missing.append(pkg)
    except ImportError:
        missing = REQUIRED_PACKAGES

    if missing:
        print("📦 正在同步并升级核心量化依赖 (严格校验版本):")
        print("   " + ", ".join(missing))

        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                *missing,
                "-q",
            ])
            print("  ✅ 所有依赖环境已同步至最新/指定版本")
        except subprocess.CalledProcessError:
            print("  ⚠️ 部分依赖安装或升级失败，系统将尝试降级运行")

        print()


auto_install()

# ============================================================
# 运行锁：防止 GitHub Actions / 手动重复触发导致重复扣费
# ============================================================

class RunLock:
    def __init__(self, lock_path: str, stale_seconds: int = 7200):
        self.lock_path = lock_path
        self.stale_seconds = stale_seconds
        self.fd = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)

        if os.path.exists(self.lock_path):
            try:
                age = time.time() - os.path.getmtime(self.lock_path)
                if age > self.stale_seconds:
                    print(f"  [LOCK] 清理过期锁: {self.lock_path}, age={age:.0f}s")
                    os.remove(self.lock_path)
            except Exception:
                pass

        try:
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode("utf-8"))
            return True
        except FileExistsError:
            return False

    def release(self):
        try:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        except Exception:
            pass

        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass


# ============================================================
# 日期逻辑
# ============================================================

def get_target_date(offset=0):
    """
    竞彩业务日逻辑。

    默认 VMAX_DATE_SHIFT_HOURS=11：
    北京时间凌晨 00:00 - 10:59 仍归入前一个竞彩业务日。

    例如北京时间 2026-05-06 04:00：
      自然日是 2026-05-06
      减 11 小时后是 2026-05-05 17:00
      程序 today = 2026-05-05

    如果以后想按自然日跑，设置：
      VMAX_DATE_SHIFT_HOURS=0
    """
    beijing_tz = timezone(timedelta(hours=8))
    shift_hours = env_int("VMAX_DATE_SHIFT_HOURS", 11)
    now = datetime.now(beijing_tz) - timedelta(hours=shift_hours)
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")


def configure_ai_defaults():
    """Production defaults: bounded batches, no debate/retry chain."""
    defaults = {
        "AI_RUN_DAYS": "today", "VMAX_RUN_DAYS": "today",
        "AI_RUN_MODE": "single_pass", "AI_PRIMARY_MODEL": "gpt",
        "AI_BATCH_SIZE": "6", "AI_CHUNK_CONCURRENCY": "2",
        "AI_MODEL_CONCURRENCY": "2", "AI_SINGLE_PASS_MAX_CALLS": "12",
        "AI_HTTP_TOTAL_TIMEOUT": "180", "AI_CONNECT_TIMEOUT": "20",
        "AI_READ_TIMEOUT": "180", "AI_PERSISTENT_CACHE_ENABLED": "true",
        "AI_CACHE_DIR": "data/ai_cache", "AI_DECISION_CACHE_TTL": "1800",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def print_runtime_config():
    print("AI运行配置（次数为每次运行的实际硬上限）:")
    for key in ("AI_RUN_MODE", "AI_PRIMARY_MODEL", "AI_BATCH_SIZE",
                "AI_CHUNK_CONCURRENCY", "AI_MODEL_CONCURRENCY",
                "AI_SINGLE_PASS_MAX_CALLS", "AI_HTTP_TOTAL_TIMEOUT",
                "AI_PERSISTENT_CACHE_ENABLED", "AI_DECISION_CACHE_TTL"):
        print(f"   {key}={os.environ.get(key, '')}")


# ============================================================
# 主流程：只跑 today
# ============================================================

def main():
    beijing_tz = timezone(timedelta(hours=8))
    now_time = datetime.now(beijing_tz)
    session = "morning" if now_time.hour < 15 else "evening"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(repo_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    lock_path = os.path.join(data_dir, ".vmax_main_today.lock")
    lock = RunLock(
        lock_path,
        stale_seconds=env_int("VMAX_MAIN_LOCK_STALE_SECONDS", 7200),
    )

    if not lock.acquire():
        print("=" * 80)
        print("⚠️ 检测到 main.py 已有运行锁，本次退出，避免重复触发 AI 扣费。")
        print(f"LOCK: {lock_path}")
        print("=" * 80)
        return

    try:
        configure_ai_defaults()

        print("=" * 80)
        print("⚽ 量化足球投研终端 vMAX 终极版（只跑今日竞彩业务日）")
        print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
        print("🔧 核心升级：仅 today + 防重复运行锁 + 持久化缓存 + AI失败不本地兜底")
        print("=" * 80)

        print_runtime_config()

        # ============================================================
        # 自学习/复盘模块
        # ============================================================

        try:
            import verify
            verify.verify_and_learn()
        except Exception as e:
            print(f"  [WARN] 自学习模块跳过或未找到数据: {e}")

        # ============================================================
        # 输出文件骨架：前端只保留 today
        # ============================================================

        final_output = {
            "update_time": now_time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "MAX-v21.0-LEAGUE",
            "scope": "today_only",
            "top4": [],
            "matches": {
                "today": []
            },
            "runtime": {
                "session": session,
                "target_mode": "today_only",
                "date_shift_hours": os.environ.get("VMAX_DATE_SHIFT_HOURS", "11"),
                "ai_run_mode": os.environ.get("AI_RUN_MODE", ""),
                "ai_batch_size": os.environ.get("AI_BATCH_SIZE", ""),
                "ai_chunk_concurrency": os.environ.get("AI_CHUNK_CONCURRENCY", ""),
                "ai_model_concurrency": os.environ.get("AI_MODEL_CONCURRENCY", ""),
                "ai_phase1_parallel": os.environ.get("AI_PHASE1_PARALLEL", ""),
                "ai_single_pass_max_calls": os.environ.get("AI_SINGLE_PASS_MAX_CALLS", "12"),
                "ai_cache_ttl": os.environ.get("AI_DECISION_CACHE_TTL", "1800"),
            },
        }

        # ============================================================
        # 只抓 today
        # ============================================================

        day_key = "today"
        target_date = get_target_date(0)

        print("\n" + "=" * 80)
        print("🕵️‍♂️ [INTEL NETWORK] Koudai 情报源已移除，跳过 91bixin 接口。")
        print("ℹ️ 当前仅使用主数据源、赔率数据、模型特征与 AI 融合逻辑。")
        print("=" * 80)

        print(f"\n{'=' * 20} 正在并发抓取并清洗 {day_key} ({target_date}) {'=' * 20}")

        try:
            from fetch_data import async_collect_all
            from predict import run_predictions
        except Exception as e:
            print("\n" + "!" * 80)
            print(f"🚨 模块导入失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        raw_data = asyncio.run(async_collect_all(target_date))

        if not raw_data or not raw_data.get("matches"):
            print(f"  [SKIP] {target_date} 暂无比赛数据，跳过 AI 推理。")

            if not env_bool("VMAX_ALLOW_EMPTY_PUBLISH", False):
                raise RuntimeError(
                    "未抓到比赛数据，默认保护上一份 predictions.json；"
                    "如确认是无赛程日，请显式设置 VMAX_ALLOW_EMPTY_PUBLISH=true。"
                )

            final_output["matches"]["today"] = []
            final_output["update_time"] = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")

            paths = publish_prediction_outputs(data_dir, target_date, session, final_output, datetime.now(beijing_tz))

            print(f"✅ 已落盘空 today 结构，快照: {paths['snapshot']}")
            return

        # 保留下游兼容字段 information
        for match in raw_data.get("matches", []):
            match["information"] = match.get("information") or {}

        use_ai = True
        print(f"  [AI ENABLED] today 将启用 AI 推理 | 比赛数={len(raw_data.get('matches', []))}")

        results, top4 = run_predictions(raw_data, use_ai=use_ai)
        from predict import _LAST_AI_RUN_METADATA
        final_output["runtime"]["ai_run"] = dict(_LAST_AI_RUN_METADATA)

        final_output["matches"]["today"] = json.loads(
            json.dumps(results, ensure_ascii=False, default=str)
        )

        if top4:
            final_output["top4"] = [
                {
                    "rank": i + 1,
                    **t,
                    "fusion_summary": "vMAX-Dynamic-Hybrid",
                }
                for i, t in enumerate(
                    json.loads(json.dumps(top4, ensure_ascii=False, default=str))
                )
            ]

        final_output["update_time"] = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")

        paths = publish_prediction_outputs(data_dir, target_date, session, final_output, datetime.now(beijing_tz))

        print(f"  ✅ today 任务完成，数据已同步至 predictions.json")
        print(f"  🧾 不可变审计快照: {paths['snapshot']}")

        print(f"\n{'=' * 80}")
        print("✅ 全链路执行成功！今日竞彩业务日预测任务完成。")
        print(f"{'=' * 80}")

    except Exception as e:
        print("\n" + "!" * 80)
        print(f"🚨 致命崩溃: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        lock.release()


if __name__ == "__main__":
    main()
