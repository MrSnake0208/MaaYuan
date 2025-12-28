import json

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


def _safe_parse_params(raw, name: str) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"{name} 参数解析失败: {raw}")
        except Exception:
            logger.exception(f"{name} 参数解析异常")
    return {}


def _extract_recognition(detail):
    if detail is None:
        return None
    if hasattr(detail, "nodes"):
        nodes = getattr(detail, "nodes", None) or []
        if nodes:
            return getattr(nodes[0], "recognition", None)
        return None
    return detail


def _get_results(detail) -> list:
    recognition = _extract_recognition(detail)
    if recognition is None:
        return []
    for attr in ("filtered_results", "filterd_results", "all_results"):
        results = getattr(recognition, attr, None)
        if results:
            return results
    return []


def _box_to_roi(box):
    if not box:
        return None
    try:
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    except Exception:
        return None


def _wait_task(context: Context, task_detail):
    if not task_detail:
        return None
    status = getattr(task_detail, "status", None)
    if status and getattr(status, "done", False):
        return task_detail
    gen_job = getattr(context.tasker, "_gen_task_job", None)
    if callable(gen_job):
        try:
            return gen_job(task_detail.task_id).wait().get()
        except Exception:
            logger.exception("等待任务完成时异常")
    return task_detail


TARGET_ORDER = ["green", "blue", "purple"]
DETECT_ROI_DEFAULT = [63, 342, 584, 672]


def _normalize_targets(raw):
    if isinstance(raw, (list, tuple)):
        requested = []
        for item in raw:
            if isinstance(item, str):
                value = item.strip().lower()
                if value:
                    requested.append(value)
        requested_set = set(requested)
        ordered = [value for value in TARGET_ORDER if value in requested_set]
        return ordered, True, requested
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value:
            return [value], False, [value]
        return [], False, []
    return [], False, []


def _offset_roi(roi, offset):
    if not roi or not offset:
        return roi
    try:
        return [
            int(roi[0]) + int(offset[0]),
            int(roi[1]) + int(offset[1]),
            int(roi[2]),
            int(roi[3]),
        ]
    except Exception:
        return roi


def _candidate_rois(roi, detect_roi):
    candidates = []
    if roi:
        candidates.append(roi)
    if detect_roi and len(detect_roi) >= 2:
        offset_roi = _offset_roi(roi, detect_roi[:2])
        if offset_roi and offset_roi not in candidates:
            candidates.append(offset_roi)
    return candidates


@AgentServer.custom_action("ChujianriShopping")
class ChujianriShopping(CustomAction):
    """
    Args:
        - target: green | blue | purple
        - target: ["green", "blue", "purple"] (按 green -> blue -> purple 顺序执行)
    """

    ENTRY_PURCHASE = "初见日-第二策略-购买"

    TARGET_CONFIG = {
        "green": {
            "pre_task": "初见日-商铺-回顶",
            "recognition": "初见日-商铺-检测绿色物品",
            "detect_roi": DETECT_ROI_DEFAULT,
        },
        "blue": {
            "pre_task": "初见日-商铺-下一页",
            "recognition": "初见日-商铺-检测蓝色物品",
            "detect_roi": DETECT_ROI_DEFAULT,
        },
        "purple": {
            "pre_task": "初见日-商铺-下一页",
            "recognition": "初见日-商铺-检测紫色物品",
            "detect_roi": DETECT_ROI_DEFAULT,
        },
    }

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = _safe_parse_params(argv.custom_action_param, "ChujianriShopping")
        raw_target = params.get("target", "green")
        targets, is_multi, requested = _normalize_targets(raw_target)
        if not targets:
            logger.warning("ChujianriShopping: 未提供有效 target 参数")
            return CustomAction.RunResult(success=False)

        if is_multi:
            invalid = [item for item in requested if item not in self.TARGET_CONFIG]
            if invalid:
                logger.warning(
                    "ChujianriShopping: 忽略无效 target: "
                    + ", ".join(sorted(set(invalid)))
                )
        else:
            if targets[0] not in self.TARGET_CONFIG:
                logger.warning(f"初见日自动购物: 无效参数 target={targets[0]}")
                return CustomAction.RunResult(success=False)

        logger.info(f"初见日自动购物: 开始处理 targets={targets}")

        for target in targets:
            config = self.TARGET_CONFIG.get(target)
            if not config:
                continue

            _wait_task(context, context.run_task(config["pre_task"]))

            img = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(config["recognition"], img)
            results = _get_results(reco_detail)

            if not results:
                logger.info(f"初见日自动购物: {target} 未识别到可购买物品")
                continue

            def _sort_key(res):
                box = getattr(res, "box", None)
                if not box:
                    return (0, 0)
                try:
                    return (int(box[1]), int(box[0]))
                except Exception:
                    return (0, 0)

            results = sorted(results, key=_sort_key)

            valid_rois = 0
            for res in results:
                roi = _box_to_roi(getattr(res, "box", None))
                if not roi:
                    continue
                valid_rois += 1

                purchase_ok = False
                for click_roi in _candidate_rois(roi, config.get("detect_roi")):
                    overrides = {
                        self.ENTRY_PURCHASE: {
                            "action": {
                                "type": "Click",
                                "param": {"target": click_roi},
                            },
                        }
                    }
                    result = context.run_task(self.ENTRY_PURCHASE, overrides)
                    result = _wait_task(context, result)
                    if (
                        result
                        and getattr(result, "status", None)
                        and result.status.succeeded
                    ):
                        purchase_ok = True
                        break
                if not purchase_ok:
                    logger.warning(f"初见日自动购物: 执行购买失败 roi={roi}")

            if valid_rois == 0:
                logger.info(f"初见日自动购物: {target} 识别到结果但无有效 roi")

        return CustomAction.RunResult(success=True)
