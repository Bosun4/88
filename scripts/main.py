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
    "aiohttp",
    "Requests>=2.32.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "deep-translator>=1.11.4",
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
    """
    这里只设置不会影响预测结构的安全默认值。

    注意：
    不再在 main.py 里强制 AI_BATCH_SIZE=4。
    不再在 main.py 里强制 AI_MODEL_CONCURRENCY=1。
    不再在 main.py 里强制 AI_PHASE1_PARALLEL=false。

    批次、并发、是否串行，应该由 predict.py 自己支持后再由 yml/env 控制。
    main.py 不再制造“看起来是4场小批次、实际没生效”的误导日志。
    """

    # 只跑 today
    os.environ.setdefault("AI_RUN_DAYS", "today")
    os.environ.setdefault("VMAX_RUN_DAYS", "today")

    # 每个模型最大请求数按运行模式给安全默认值：fast_batch 负责全量初筛，deep_research 保持保守。
    if os.environ.get("AI_RUN_MODE", "").strip().lower() == "fast_batch":
        os.environ.setdefault("AI_MAX_REQUESTS_PER_AI", "3")
        os.environ.setdefault("AI_CHUNK_CONCURRENCY", "3")
        os.environ.setdefault("AI_MODEL_CONCURRENCY", "4")
        os.environ.setdefault("AI_PHASE1_PARALLEL", "true")
    else:
        os.environ.setdefault("AI_MAX_REQUESTS_PER_AI", "1")

    # Claude 条件触发：如果 predict.py 支持，就会生效；不支持也不会影响 main.py
    os.environ.setdefault("AI_RUN_CLAUDE_ONLY_IF_PHASE1_VALID", "true")
    os.environ.setdefault("AI_MIN_PHASE1_VALID_FOR_CLAUDE", "2")

    # Claude 终审压缩：如果 predict.py 支持，就会生效
    os.environ.setdefault("AI_USE_COMPACT_CLAUDE_AUDIT", "true")
    os.environ.setdefault("AI_MAX_PHASE1_REASON_CHARS_FOR_CLAUDE", "350")

    # 持久化缓存，避免同批重复扣费
    os.environ.setdefault("AI_PERSISTENT_CACHE_ENABLED", "true")
    os.environ.setdefault("AI_CACHE_DIR", "data/ai_cache")
    os.environ.setdefault("AI_CACHE_STRIP_VOLATILE_KEYS", "true")
    os.environ.setdefault("AI_DISK_LOCK_WAIT_SECONDS", "900")
    os.environ.setdefault("AI_DISK_LOCK_POLL_SECONDS", "3")
    os.environ.setdefault("AI_DECISION_CACHE_TTL", "1800")
    os.environ.setdefault("AI_WRITE_BATCH_RESULT_IMMEDIATELY", "true")


def print_runtime_config():
    print("⚙️ AI运行配置:")
    print(f"   VMAX_DATE_SHIFT_HOURS={os.environ.get('VMAX_DATE_SHIFT_HOURS', '11')}")
    print(f"   VMAX_RUN_DAYS={os.environ.get('VMAX_RUN_DAYS', 'today')}")
    print(f"   AI_RUN_DAYS={os.environ.get('AI_RUN_DAYS', 'today')}")
    print(f"   AI_MAX_REQUESTS_PER_AI={os.environ.get('AI_MAX_REQUESTS_PER_AI', '1')}")
    print(f"   AI_RUN_MODE={os.environ.get('AI_RUN_MODE', '')}")
    print(f"   AI_CHUNK_CONCURRENCY={os.environ.get('AI_CHUNK_CONCURRENCY', '')}")
    print(f"   AI_MODEL_CONCURRENCY={os.environ.get('AI_MODEL_CONCURRENCY', '')}")
    print(f"   AI_PHASE1_PARALLEL={os.environ.get('AI_PHASE1_PARALLEL', '')}")
    print(f"   AI_RUN_CLAUDE_ONLY_IF_PHASE1_VALID={os.environ.get('AI_RUN_CLAUDE_ONLY_IF_PHASE1_VALID', 'true')}")
    print(f"   AI_MIN_PHASE1_VALID_FOR_CLAUDE={os.environ.get('AI_MIN_PHASE1_VALID_FOR_CLAUDE', '2')}")
    print(f"   AI_USE_COMPACT_CLAUDE_AUDIT={os.environ.get('AI_USE_COMPACT_CLAUDE_AUDIT', 'true')}")
    print(f"   AI_PERSISTENT_CACHE_ENABLED={os.environ.get('AI_PERSISTENT_CACHE_ENABLED', 'true')}")
    print(f"   AI_DECISION_CACHE_TTL={os.environ.get('AI_DECISION_CACHE_TTL', '1800')}")
    print(f"   AI_CACHE_DIR={os.environ.get('AI_CACHE_DIR', 'data/ai_cache')}")

    if os.environ.get("AI_BATCH_SIZE"):
        print(f"   AI_BATCH_SIZE={os.environ.get('AI_BATCH_SIZE')}  # 来自 yml/env，main.py 未强制设置")
    else:
        print("   AI_BATCH_SIZE=<unset>  # main.py 未强制设置")

    if os.environ.get("AI_MODEL_CONCURRENCY"):
        print(f"   AI_MODEL_CONCURRENCY={os.environ.get('AI_MODEL_CONCURRENCY')}  # 来自 yml/env，main.py 未强制设置")
    else:
        print("   AI_MODEL_CONCURRENCY=<unset>  # main.py 未强制设置")

    if os.environ.get("AI_PHASE1_PARALLEL"):
        print(f"   AI_PHASE1_PARALLEL={os.environ.get('AI_PHASE1_PARALLEL')}  # 来自 yml/env，main.py 未强制设置")
    else:
        print("   AI_PHASE1_PARALLEL=<unset>  # main.py 未强制设置")


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

        target_path = os.path.join(data_dir, "predictions.json")

        final_output = {
            "update_time": now_time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "MAX-v1.1",
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
                "ai_max_requests_per_ai": os.environ.get("AI_MAX_REQUESTS_PER_AI", "1"),
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

            write_json_atomic(target_path, final_output)

            history_path = os.path.join(
                data_dir,
                f"history_{target_date}_today_{session}.json",
            )
            write_json_atomic(history_path, final_output)

            print("✅ 已落盘空 today 结构。")
            return

        # 保留下游兼容字段 information
        for match in raw_data.get("matches", []):
            match["information"] = match.get("information") or {}

        use_ai = True
        print(f"  [AI ENABLED] today 将启用 AI 推理 | 比赛数={len(raw_data.get('matches', []))}")

        results, top4 = run_predictions(raw_data, use_ai=use_ai)

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

        write_json_atomic(target_path, final_output)

        history_path = os.path.join(
            data_dir,
            f"history_{target_date}_today_{session}.json",
        )
        write_json_atomic(history_path, final_output)

        print(f"  ✅ today 任务完成，数据已同步至 predictions.json")

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
