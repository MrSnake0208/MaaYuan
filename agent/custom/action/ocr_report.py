import json
from datetime import datetime

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


def _parse_params(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"OcrReport: invalid params: {raw}")
            return {}
        if isinstance(data, dict):
            return data
        logger.warning(f"OcrReport: params type invalid: {type(data)}")
        return {}
    logger.warning(f"OcrReport: params type invalid: {type(raw)}")
    return {}


def _format_box(box):
    if not box:
        return None
    try:
        return [int(v) for v in box]
    except Exception:
        return box


def _format_result(result) -> dict:
    if result is None:
        return {}
    payload = {}
    box = getattr(result, "box", None)
    if box is not None:
        payload["box"] = _format_box(box)
    for attr in ("text", "detail", "score", "count", "label", "cls_index"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value is not None:
                payload[attr] = value
    return payload


def _extract_text_from_detail(detail, depth: int = 0) -> str:
    if depth > 4:
        return ""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        if "text" in detail and detail["text"] is not None:
            return str(detail["text"])
        for key in ("best", "raw_detail", "detail"):
            if key in detail:
                text = _extract_text_from_detail(detail.get(key), depth + 1)
                if text:
                    return text
        for key in ("filtered", "all"):
            items = detail.get(key)
            if isinstance(items, list):
                for item in items:
                    text = _extract_text_from_detail(item, depth + 1)
                    if text:
                        return text
    return ""


def _get_best_text(detail) -> str:
    if detail is None:
        return ""
    best = getattr(detail, "best_result", None)
    if best is not None:
        if hasattr(best, "text") and getattr(best, "text", None) is not None:
            return str(best.text)
        text = _extract_text_from_detail(getattr(best, "detail", None))
        if text:
            return text
    text = _extract_text_from_detail(getattr(detail, "raw_detail", None))
    if text:
        return text
    for results in (
        getattr(detail, "filterd_results", None),
        getattr(detail, "filtered_results", None),
        getattr(detail, "all_results", None),
    ):
        if results:
            for res in results:
                if hasattr(res, "text") and getattr(res, "text", None) is not None:
                    return str(res.text)
                text = _extract_text_from_detail(getattr(res, "detail", None))
                if text:
                    return text
    return ""


@AgentServer.custom_action("OcrReport")
class OcrReport(CustomAction):
    """
    Print recognition results for a specified node.

    Args:
        - recognition: recognition node name
        - format: custom output format, use {result} for best text
        - export: whether to append formatted output to a text file
        - filename: output file name for export
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = _parse_params(argv.custom_action_param)
        node = params.get("recognition")
        if isinstance(node, str):
            node = node.strip()
        if not node:
            logger.warning("OcrReport: missing recognition param")
            return CustomAction.RunResult(success=False)

        img = context.tasker.controller.post_screencap().wait().get()
        detail = context.run_recognition(node, img)
        if not detail:
            logger.info(f"OcrReport: {node} returned no detail")
            return CustomAction.RunResult(success=True)

        # hit = bool(getattr(detail, "box", None) or getattr(detail, "best_result", None))
        # logger.info(f"OcrReport: node={node}, algorithm={detail.algorithm}, hit={hit}")

        # best_result = getattr(detail, "best_result", None)
        # best = _format_result(best_result)
        # if best:
        #     logger.info("OcrReport: best=" + json.dumps(best, ensure_ascii=False))
        fmt = params.get("format")
        if isinstance(fmt, str):
            text = _get_best_text(detail)
            formatted = fmt.replace("{result}", text)
            logger.info(formatted)

            if params.get("export"):
                filename = params.get("filename")
                if isinstance(filename, str) and filename.strip():
                    file_path = filename.strip()
                    if not file_path.lower().endswith(".txt"):
                        file_path += ".txt"
                    try:
                        with open(file_path, "a", encoding="utf-8") as f:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f"{timestamp} {formatted}\n")
                    except Exception:
                        logger.exception(f"OcrReport: failed to write {file_path}")
                else:
                    logger.warning("OcrReport: export enabled but filename is empty")

        results = (
            getattr(detail, "filterd_results", None)
            or getattr(detail, "filtered_results", None)
            or getattr(detail, "all_results", None)
        )
        if results:
            for idx, res in enumerate(results):
                payload = _format_result(res)
                # logger.info(
                #     f"OcrReport: result[{idx}]="
                #     + json.dumps(payload, ensure_ascii=False)
                # )
        return CustomAction.RunResult(success=True)
