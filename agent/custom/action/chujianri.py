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


@AgentServer.custom_action("ChujianriShopping")
class ChujianriShopping(CustomAction):
    """
    Args:
        - target: green | blue | purple
    """

    ENTRY_PURCHASE = "初见日-第二策略-购买"

    TARGET_CONFIG = {
        "green": {
            "pre_task": "初见日-商铺-回顶",
            "recognition": "初见日-商铺-检测绿色物品",
        },
        "blue": {
            "pre_task": "初见日-商铺-下一页",
            "recognition": "初见日-商铺-检测蓝色物品",
        },
        "purple": {
            "pre_task": "初见日-商铺-下一页",
            "recognition": "初见日-商铺-检测紫色物品",
        },
    }

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = _safe_parse_params(argv.custom_action_param, "ChujianriShopping")
        target = str(params.get("target", "green") or "").strip().lower()
        config = self.TARGET_CONFIG.get(target)
        if not config:
            logger.warning(f"ChujianriShopping: 无效参数 target={target}")
            return CustomAction.RunResult(success=False)

        logger.info(f"ChujianriShopping: 开始处理 target={target}")
        _wait_task(context, context.run_task(config["pre_task"]))

        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(config["recognition"], img)
        results = _get_results(reco_detail)

        if not results:
            logger.info("ChujianriShopping: 未识别到可购买物品")
            return CustomAction.RunResult(success=True)

        executed = 0
        for res in results:
            roi = _box_to_roi(getattr(res, "box", None))
            if not roi:
                continue

            overrides = {
                self.ENTRY_PURCHASE: {
                    "action": {"type": "Click", "param": {"target": roi}},
                }
            }
            result = context.run_task(self.ENTRY_PURCHASE, overrides)
            result = _wait_task(context, result)
            executed += 1

            if (
                result
                and getattr(result, "status", None)
                and not result.status.succeeded
            ):
                logger.warning(f"ChujianriShopping: 执行购买失败 roi={roi}")

        if executed == 0:
            logger.info("ChujianriShopping: 识别到结果但无有效 roi")

        return CustomAction.RunResult(success=True)
