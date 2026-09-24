"""Development-only separation of local candidates and sustained visual evidence."""

from .contracts import Prediction, mask_events

FUSION_VERSION = "independent-specialists-v1"
EVIDENCE_VERSION = "observed-evidence-v1"
SUSTAINED_PROMPT = (
    "这是每日北京时间15点观测。"
    "你负责独立分析整窗的持续变化、台阶和异常波动，不接收其他检测器的结果。"
    "孤立的单日离群由另一独立分支定位，本分支不输出仅单日的孤立尖峰；"
    "但不能因此忽略台阶起点或持续事件中的局部波动。"
    "逐分量检查是否相对此前可观测状态发生持续水平改变、速率改变、加速或波动增大。"
    "稳定的背景斜率不自动算异常，但没有突变不能作为排除持续形变的理由。"
    "不存在已认证的正常参考段，不假定前45日正常；证据不足可拒判，不能推断物理成因。"
    "定位的是受影响位移区间：区分增长过程与增长结束后仍保留的偏移；"
    "若位移未恢复，区间可延续到窗口末尾，不能只标斜率较大的一段。"
    "依据观测确定分量和边界，不因三轴同图就自动扩展到其他分量。"
    "缺口不是恢复或停止形变的证据，也不为缺口内部补造证据。"
    "可输出空事件；无法判断时返回insufficient。"
)
EVIDENCE_PROMPT = (
    "这是每日北京时间15点观测。你独立分析整窗的持续变化、台阶和异常波动，"
    "不接收其他检测器结果；孤立单日尖峰由另一分支处理。"
    "逐一检查N、E、U，不因同图出现变化就给其他分量添加事件。"
    "每个报告的事件必须在reason中说明：所选分量、起止索引及对应日期、"
    "起点前后或变化过程中的实际有效观测依据，以及结束边界的依据。"
    "只引用图和元数据能确认的观测；无法读出精确数值时描述相对水平或斜率，不编造数值。"
    "将新发生的变化过程与其后保留的位移分别描述：若位移没有恢复，"
    "受影响区间可以延续到窗末，不能机械截短；若后续观测不支持延续，则在有证据处结束。"
    "稳定的既有斜率、图像尺度造成的视觉印象和缺测本身都不能单独构成变化证据。"
    "缺口内不补造观测；边界不清时在reason中说明不确定性，不把不确定扩成整轴整窗。"
    "没有已认证的正常参考段，不假定前45日正常，不推断仪器故障或滑坡成因。"
    "若没有足够依据确认事件，输出空events；数据不足以判断时返回insufficient。"
)


def merge_branches(local: Prediction, visual: Prediction, length: int) -> Prediction:
    """Union only completed branches; retain raw, attributed events in tool traces."""
    if visual.status != "ok":
        return Prediction(
            status=visual.status, reason="independent visual branch: " + visual.reason
        )
    if local.status != "ok":
        return Prediction(status=local.status, reason="local branch: " + local.reason)
    merged = local.mask(length) | visual.mask(length)
    return Prediction(
        events=mask_events(merged, kind="combined_candidate"),
        reason="Independent branch union; original event kinds and sources are in tool traces.",
    )
