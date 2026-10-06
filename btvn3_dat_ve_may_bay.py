# -*- coding: utf-8 -*-
"""BTVN#3 · Agent Đặt Vé Máy Bay — Ba mẫu thiết kế & Đánh giá.

Cài đặt ba mẫu thiết kế agent (ReAct, Plan-then-Execute, Hybrid) cho bài toán
đặt vé máy bay, sử dụng model giả lập và bộ tool mockup.

Chạy:
    python btvn3_dat_ve_may_bay.py --agent react
    python btvn3_dat_ve_may_bay.py --agent plan
    python btvn3_dat_ve_may_bay.py --agent hybrid
    python btvn3_dat_ve_may_bay.py --all          # chạy tất cả + bảng đánh giá
    python btvn3_dat_ve_may_bay.py --tc 1          # chỉ chạy test case 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from lib.tools_maybay import (
    ALL_TOOLS, reset_data, tools_langchain,
    search_flights, get_flight_details, check_seat_availability,
    book_flight, cancel_booking, get_booking_status,
)
from lib.harness_maybay import (
    DataConstraintChecker, CompletionChecker, PermissionChecker,
    HandoffReporter, FlightLoopDetector, FlightFactChecker,
)

# ============================================================ TEST CASES
TEST_CASES = {
    "tc1": {
        "name": "Đặt vé bình thường (happy path)",
        "question": "Tôi muốn đặt vé máy bay từ Hà Nội đi TP.HCM ngày 20/12/2024, "
                    "hạng economy, tên Nguyen Van A, thanh toán thẻ tín dụng.",
        "user": "user_a",
        "expect_success": True,
        "expect_booking": True,
    },
    "tc2": {
        "name": "Chuyến bay không tồn tại",
        "question": "Tìm chuyến bay từ Hà Nội đi Phú Quốc ngày 25/12/2024.",
        "user": "user_a",
        "expect_success": True,
        "expect_booking": False,
    },
    "tc3": {
        "name": "Hết ghế, cần chuyển chuyến",
        "question": "Đặt vé chuyến VN101 hạng economy, tên Nguyen Van A, "
                    "thanh toán thẻ tín dụng.",
        "user": "user_a",
        "expect_success": True,
        "expect_booking": True,   # chuyển sang VN100
    },
    "tc4": {
        "name": "Huỷ vé không có quyền",
        "question": "Huỷ booking BK002.",
        "user": "user_a",
        "expect_success": False,
        "expect_booking": False,
    },
    "tc5": {
        "name": "Đặt business khi không phải VIP",
        "question": "Đặt vé chuyến VJ200 hạng business, tên Nguyen Van A, "
                    "thanh toán ewallet.",
        "user": "user_a",
        "expect_success": False,
        "expect_booking": False,
    },
}


# ================================================ MODEL GIẢ LẬP MÁY BAY
class ModelGiaMayBay(BaseChatModel):
    """Model giả lập cho agent đặt vé máy bay.

    Tham số:
        kich_ban:  mã test case ("tc1" – "tc5")
        kieu:      mẫu agent ("react", "plan", "hybrid")
    """
    kich_ban: str = "tc1"
    kieu: str = "react"
    _luot: int = 0

    @property
    def _llm_type(self) -> str:
        return "btvn3-model-gia-maybay"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModelGiaMayBay":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._luot += 1
        ai = self._quyet_dinh(messages)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _quyet_dinh(self, messages: list[BaseMessage]) -> AIMessage:
        if self.kieu == "react":
            return self._react(messages)
        elif self.kieu == "plan":
            return self._plan_execute(messages)
        else:  # hybrid
            return self._hybrid(messages)

    # ────────────────────────────────────── ReAct
    def _react(self, messages: list[BaseMessage]) -> AIMessage:
        tool_obs = self._get_observations(messages)
        called = self._get_called_tools(messages)

        if self.kich_ban == "tc1":
            return self._react_tc1(tool_obs, called)
        elif self.kich_ban == "tc2":
            return self._react_tc2(tool_obs, called)
        elif self.kich_ban == "tc3":
            return self._react_tc3(tool_obs, called)
        elif self.kich_ban == "tc4":
            return self._react_tc4(tool_obs, called)
        elif self.kich_ban == "tc5":
            return self._react_tc5(tool_obs, called)
        return AIMessage(content="Không hiểu yêu cầu.")

    def _react_tc1(self, obs, called):
        if not any(c[0] == "search_flights" for c in called):
            return self._goi("search_flights",
                             {"departure": "HAN", "destination": "SGN", "date": "2024-12-20"})
        if not any(c[0] == "check_seat_availability" for c in called):
            return self._goi("check_seat_availability",
                             {"flight_id": "VN100", "seat_class": "economy"})
        if not any(c[0] == "book_flight" for c in called):
            return self._goi("book_flight",
                             {"flight_id": "VN100", "passenger_name": "Nguyen Van A",
                              "seat_class": "economy", "payment_method": "credit_card"})
        # Tìm booking_id từ observation
        bk_id = self._find_in_obs(obs, "booking_id") or "BK003"
        return AIMessage(
            content=f"Đặt vé thành công! Mã đặt chỗ: {bk_id}. "
                    f"Chuyến VN100, Hà Nội → TP.HCM, ngày 2024-12-20, "
                    f"khởi hành 06:00, giá 1.500.000đ, hạng economy.")

    def _react_tc2(self, obs, called):
        if not any(c[0] == "search_flights" for c in called):
            return self._goi("search_flights",
                             {"departure": "HAN", "destination": "PQC", "date": "2024-12-25"})
        return AIMessage(
            content="Không tìm thấy chuyến bay từ HAN đến PQC ngày 2024-12-25. "
                    "Vui lòng thử ngày khác hoặc tuyến bay khác.")

    def _react_tc3(self, obs, called):
        seat_checks = [c for c in called if c[0] == "check_seat_availability"]
        if not seat_checks:
            return self._goi("check_seat_availability",
                             {"flight_id": "VN101", "seat_class": "economy"})
        # Nếu đã check VN101, check VN100
        checked_flights = {c[1].get("flight_id") for c in seat_checks}
        if "VN101" in checked_flights and "VN100" not in checked_flights:
            return self._goi("check_seat_availability",
                             {"flight_id": "VN100", "seat_class": "economy"})
        if not any(c[0] == "book_flight" for c in called):
            return self._goi("book_flight",
                             {"flight_id": "VN100", "passenger_name": "Nguyen Van A",
                              "seat_class": "economy", "payment_method": "credit_card"})
        bk_id = self._find_in_obs(obs, "booking_id") or "BK003"
        return AIMessage(
            content=f"Chuyến VN101 đã hết ghế economy. Đã chuyển sang VN100. "
                    f"Đặt vé thành công! Mã: {bk_id}, chuyến VN100, giá 1.500.000đ.")

    def _react_tc4(self, obs, called):
        if not any(c[0] == "cancel_booking" for c in called):
            return self._goi("cancel_booking", {"booking_id": "BK002"})
        return AIMessage(
            content="Không thể huỷ booking BK002. Bạn không có quyền huỷ booking "
                    "này vì nó thuộc về người dùng khác. Vui lòng liên hệ nhân viên hỗ trợ.")

    def _react_tc5(self, obs, called):
        if not any(c[0] == "book_flight" for c in called):
            return self._goi("book_flight",
                             {"flight_id": "VJ200", "passenger_name": "Nguyen Van A",
                              "seat_class": "business", "payment_method": "ewallet"})
        return AIMessage(
            content="Không thể đặt hạng business. Chỉ thành viên VIP mới được đặt "
                    "hạng business. Bạn có thể đặt hạng economy hoặc nâng cấp tài khoản VIP.")

    # ────────────────────────────────── Plan-then-Execute
    def _plan_execute(self, messages: list[BaseMessage]) -> AIMessage:
        tool_obs = self._get_observations(messages)
        called = self._get_called_tools(messages)

        # Phase 1: Sinh kế hoạch + bước đầu tiên (tool call đi kèm)
        if self._luot == 1:
            return self._gen_plan_with_first_step()

        # Phase 2: Thực hiện tuần tự các bước còn lại
        if self.kich_ban == "tc1":
            return self._react_tc1(tool_obs, called)
        elif self.kich_ban == "tc2":
            return self._react_tc2(tool_obs, called)
        elif self.kich_ban == "tc3":
            return self._react_tc3(tool_obs, called)
        elif self.kich_ban == "tc4":
            return self._react_tc4(tool_obs, called)
        elif self.kich_ban == "tc5":
            return self._react_tc5(tool_obs, called)
        return AIMessage(content="Không hiểu yêu cầu.")

    def _gen_plan_with_first_step(self) -> AIMessage:
        """Sinh kế hoạch kèm tool call đầu tiên để create_agent không dừng."""
        plans_and_calls = {
            "tc1": (
                "KẾ HOẠCH ĐẶT VÉ:\n"
                "Bước 1: Tìm chuyến bay HAN → SGN ngày 2024-12-20\n"
                "Bước 2: Kiểm tra ghế trống hạng economy\n"
                "Bước 3: Đặt vé cho Nguyen Van A, thanh toán credit_card\n"
                "Bước 4: Xác nhận và trả kết quả\n"
                "→ Thực hiện Bước 1:",
                "search_flights",
                {"departure": "HAN", "destination": "SGN", "date": "2024-12-20"},
            ),
            "tc2": (
                "KẾ HOẠCH TÌM CHUYẾN BAY:\n"
                "Bước 1: Tìm chuyến bay HAN → PQC ngày 2024-12-25\n"
                "Bước 2: Trả kết quả tìm kiếm\n"
                "→ Thực hiện Bước 1:",
                "search_flights",
                {"departure": "HAN", "destination": "PQC", "date": "2024-12-25"},
            ),
            "tc3": (
                "KẾ HOẠCH ĐẶT VÉ:\n"
                "Bước 1: Kiểm tra ghế chuyến VN101 hạng economy\n"
                "Bước 2: Nếu hết, tìm chuyến thay thế\n"
                "Bước 3: Đặt vé chuyến có ghế trống\n"
                "Bước 4: Xác nhận kết quả\n"
                "→ Thực hiện Bước 1:",
                "check_seat_availability",
                {"flight_id": "VN101", "seat_class": "economy"},
            ),
            "tc4": (
                "KẾ HOẠCH HUỶ VÉ:\n"
                "Bước 1: Huỷ booking BK002\n"
                "Bước 2: Xác nhận kết quả\n"
                "→ Thực hiện Bước 1:",
                "cancel_booking",
                {"booking_id": "BK002"},
            ),
            "tc5": (
                "KẾ HOẠCH ĐẶT VÉ BUSINESS:\n"
                "Bước 1: Đặt vé chuyến VJ200 hạng business\n"
                "Bước 2: Xác nhận kết quả\n"
                "→ Thực hiện Bước 1:",
                "book_flight",
                {"flight_id": "VJ200", "passenger_name": "Nguyen Van A",
                 "seat_class": "business", "payment_method": "ewallet"},
            ),
        }
        plan, tool_name, tool_args = plans_and_calls.get(
            self.kich_ban,
            ("KẾ HOẠCH: Thực hiện yêu cầu.", "search_flights",
             {"departure": "HAN", "destination": "SGN", "date": "2024-12-20"}),
        )
        return AIMessage(content=plan, tool_calls=[
            {"name": tool_name, "args": tool_args,
             "id": f"plan_{self._luot}", "type": "tool_call"}
        ])

    # ──────────────────────────────────────── Hybrid
    def _hybrid(self, messages: list[BaseMessage]) -> AIMessage:
        tool_obs = self._get_observations(messages)
        called = self._get_called_tools(messages)

        # Phase 1: Kế hoạch ban đầu + bước 1
        if self._luot == 1:
            return self._gen_plan_with_first_step()

        # Phase 2: Đang chạy mà gặp trở ngại (hết ghế / lỗi) → Re-plan linh hoạt
        last_err = self._last_error(obs=tool_obs)
        has_replan = any(
            getattr(m, "type", None) == "ai"
            and "ĐIỀU CHỈNH" in str(getattr(m, "content", ""))
            for m in messages
        )
        if last_err and not has_replan:
            return self._gen_replan_with_step(last_err)

        # Phase 3: Thực hiện các bước tiếp theo
        if self.kich_ban == "tc1":
            return self._react_tc1(tool_obs, called)
        elif self.kich_ban == "tc2":
            return self._react_tc2(tool_obs, called)
        elif self.kich_ban == "tc3":
            return self._hybrid_tc3_continue(tool_obs, called)
        elif self.kich_ban == "tc4":
            return self._react_tc4(tool_obs, called)
        elif self.kich_ban == "tc5":
            return self._react_tc5(tool_obs, called)
        return AIMessage(content="Không hiểu yêu cầu.")

    def _gen_replan_with_step(self, error: str) -> AIMessage:
        """Sinh bước điều chỉnh kế hoạch kèm tool call tiếp theo."""
        if self.kich_ban == "tc3":
            return AIMessage(
                content="ĐIỀU CHỈNH KẾ HOẠCH:\n"
                        f"Phát hiện: {error}.\n"
                        "Bước thay thế 1: Kiểm tra ghế chuyến VN100 hạng economy\n"
                        "Bước thay thế 2: Đặt vé chuyến VN100 nếu còn ghế\n"
                        "→ Thực hiện bước thay thế 1:",
                tool_calls=[{
                    "name": "check_seat_availability",
                    "args": {"flight_id": "VN100", "seat_class": "economy"},
                    "id": f"replan_{self._luot}",
                    "type": "tool_call",
                }],
            )
        return AIMessage(
            content=f"ĐIỀU CHỈNH KẾ HOẠCH:\nLỗi: {error}\nBàn giao cho người dùng.",
        )

    def _hybrid_tc3_continue(self, obs, called) -> AIMessage:
        if not any(c[0] == "book_flight" for c in called):
            return self._goi("book_flight",
                             {"flight_id": "VN100", "passenger_name": "Nguyen Van A",
                              "seat_class": "economy", "payment_method": "credit_card"})
        bk_id = self._find_in_obs(obs, "booking_id") or "BK003"
        return AIMessage(
            content=f"Chuyến VN101 đã hết ghế economy. Đã điều chỉnh kế hoạch chuyển sang VN100. "
                    f"Đặt vé thành công! Mã: {bk_id}, chuyến VN100, giá 1.500.000đ.")

    # ──────────────────────────────── tiện ích chung
    def _goi(self, name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call_{self._luot}", "type": "tool_call"}
        ])

    @staticmethod
    def _get_observations(messages: list[BaseMessage]) -> list[dict]:
        obs = []
        for m in messages:
            if getattr(m, "type", None) == "tool":
                try:
                    obs.append(json.loads(m.content))
                except Exception:
                    obs.append({"raw": m.content})
        return obs

    @staticmethod
    def _get_called_tools(messages: list[BaseMessage]) -> list[tuple]:
        called = []
        for m in messages:
            for c in getattr(m, "tool_calls", None) or []:
                called.append((c["name"], c.get("args", {})))
        return called

    @staticmethod
    def _find_in_obs(obs: list[dict], key: str) -> str | None:
        for o in reversed(obs):
            if key in o:
                return str(o[key])
        return None

    @staticmethod
    def _last_error(obs: list[dict]) -> str | None:
        for o in reversed(obs):
            if isinstance(o, dict):
                if o.get("status") == "error":
                    return o.get("message", o.get("error", "unknown"))
                if o.get("available") == 0:
                    return f"Hết ghế {o.get('seat_class', '')} trên chuyến {o.get('flight_id', '')}"
        return None


# ============================================ MIDDLEWARE HARNESS
from langchain.agents import create_agent                                    # noqa: E402
from langchain.agents.middleware import (                                    # noqa: E402
    AgentMiddleware,
    ModelCallLimitMiddleware,
    hook_config,
)


class FlightHarnessMiddleware(AgentMiddleware):
    """Middleware tổng hợp: kiểm dữ liệu, kiểm quyền, phát hiện lặp, bàn giao.

    Kết hợp tất cả lớp harness thành một middleware duy nhất để cắm vào
    create_agent.
    """

    def __init__(self):
        super().__init__()
        self.loop_det = FlightLoopDetector(window=6, repeat_k=2)
        self.actions_tried: list[str] = []
        self.tool_results: list[str] = []

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            # Model trả lời cuối cùng — kiểm căn cứ
            answer = str(last.content)
            # Bỏ qua plan/replan messages (không phải câu trả lời cuối)
            if any(kw in answer for kw in ("KẾ HOẠCH", "ĐIỀU CHỈNH")):
                return None
            if answer.strip() and self.tool_results:
                fact_result = FlightFactChecker.check(answer, self.tool_results)
                if not fact_result["dat"]:
                    safe = FlightFactChecker.safe_answer(answer, self.tool_results)
                    report = HandoffReporter.create(
                        stop_reason="Câu trả lời chứa dữ kiện không có căn cứ từ tool",
                        actions_tried=self.actions_tried,
                        booking_state={"fabricated": fact_result["khong_co_nguon"]},
                        question="Kiểm tra lại thông tin và bổ sung nguồn dữ liệu.",
                    )
                    return {
                        "jump_to": "end",
                        "messages": [AIMessage(content=HandoffReporter.format(report))],
                    }
            return None

        for c in last.tool_calls:
            name, args = c["name"], c["args"]
            action_str = f"{name}({args})"
            self.actions_tried.append(action_str)

            # 1. Kiểm ràng buộc dữ liệu
            data_errors = self._check_data(name, args)
            if data_errors:
                report = HandoffReporter.create(
                    stop_reason=f"Dữ liệu không hợp lệ: {'; '.join(data_errors)}",
                    actions_tried=self.actions_tried,
                    booking_state={"validation_errors": data_errors},
                    question="Sửa lại tham số và thử lại.",
                )
                return {
                    "jump_to": "end",
                    "messages": [AIMessage(content=HandoffReporter.format(report))],
                }

            # 2. Kiểm quyền
            perm_error = self._check_permission(name, args)
            if perm_error:
                report = HandoffReporter.create(
                    stop_reason=perm_error,
                    actions_tried=self.actions_tried,
                    booking_state={"permission": "denied"},
                    question="Liên hệ nhân viên hỗ trợ hoặc đổi hành động.",
                )
                return {
                    "jump_to": "end",
                    "messages": [AIMessage(content=HandoffReporter.format(report))],
                }

            # 3. Phát hiện lặp
            loop_warn = self.loop_det.check(name, args, progress=0)
            if loop_warn:
                report = HandoffReporter.create(
                    stop_reason=loop_warn,
                    actions_tried=self.actions_tried,
                    booking_state={"tool_calls": len(self.actions_tried), "success": 0},
                    question="Agent bị lặp. Cần can thiệp thủ công.",
                )
                return {
                    "jump_to": "end",
                    "messages": [AIMessage(content=HandoffReporter.format(report))],
                }

        return None

    def after_tool(self, state, runtime):
        """Lưu kết quả tool để kiểm căn cứ sau này."""
        last = state["messages"][-1]
        if getattr(last, "type", None) == "tool":
            self.tool_results.append(str(last.content))
        return None

    @staticmethod
    def _check_data(name: str, args: dict) -> list[str]:
        if name == "search_flights":
            return DataConstraintChecker.validate_search(
                args.get("departure", ""),
                args.get("destination", ""),
                args.get("date", ""))
        if name == "book_flight":
            return DataConstraintChecker.validate_booking(
                args.get("flight_id", ""),
                args.get("passenger_name", ""),
                args.get("seat_class", ""),
                args.get("payment_method", ""))
        return []

    @staticmethod
    def _check_permission(name: str, args: dict) -> str | None:
        uid = os.environ.get("BTVN3_USER", "user_a")
        if name == "cancel_booking":
            result = PermissionChecker.can_cancel(uid, args.get("booking_id", ""))
            if not result["allowed"]:
                return result["reason"]
        if name == "book_flight":
            result = PermissionChecker.check_all(
                uid, "book", seat_class=args.get("seat_class", ""))
            if not result["allowed"]:
                return result["reason"]
        return None


# ========================================== CHẠY AGENT
def run_react(tc_id: str, verbose: bool = True) -> dict:
    """Chạy ReAct agent cho một test case."""
    tc = TEST_CASES[tc_id]
    reset_data()
    os.environ["BTVN3_USER"] = tc["user"]

    model = ModelGiaMayBay(kich_ban=tc_id, kieu="react")
    tools = tools_langchain()
    middleware = [
        FlightHarnessMiddleware(),
        ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
    ]

    agent = create_agent(model=model, tools=tools, middleware=middleware)

    if verbose:
        print(f"\n{'='*70}")
        print(f"ReAct · {tc_id}: {tc['name']}")
        print(f"{'='*70}")

    start = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": tc["question"]}]},
            {"recursion_limit": 30})
    except Exception as e:
        elapsed = time.time() - start
        if verbose:
            print(f"  LỖI: {type(e).__name__}: {str(e)[:100]}")
        return _make_result(tc_id, "react", [], elapsed, error=str(e))

    elapsed = time.time() - start
    messages = result["messages"]
    if verbose:
        _print_trace(messages)

    return _make_result(tc_id, "react", messages, elapsed)


def run_plan_execute(tc_id: str, verbose: bool = True) -> dict:
    """Chạy Plan-then-Execute agent.

    Phase 1: Model sinh kế hoạch (AIMessage text, không gọi tool)
    Phase 2: Thực hiện tuần tự theo kế hoạch qua create_agent
    """
    tc = TEST_CASES[tc_id]
    reset_data()
    os.environ["BTVN3_USER"] = tc["user"]

    model = ModelGiaMayBay(kich_ban=tc_id, kieu="plan")
    tools = tools_langchain()
    harness = FlightHarnessMiddleware()
    middleware = [
        harness,
        ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
    ]

    agent = create_agent(model=model, tools=tools, middleware=middleware)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Plan-then-Execute · {tc_id}: {tc['name']}")
        print(f"{'='*70}")

    start = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": tc["question"]}]},
            {"recursion_limit": 30})
    except Exception as e:
        elapsed = time.time() - start
        if verbose:
            print(f"  LỖI: {type(e).__name__}: {str(e)[:100]}")
        return _make_result(tc_id, "plan", [], elapsed, error=str(e))

    elapsed = time.time() - start
    messages = result["messages"]
    if verbose:
        _print_trace(messages)

    return _make_result(tc_id, "plan", messages, elapsed)


def run_hybrid(tc_id: str, verbose: bool = True) -> dict:
    """Chạy Hybrid agent: Plan → Execute → Re-plan nếu gặp lỗi."""
    tc = TEST_CASES[tc_id]
    reset_data()
    os.environ["BTVN3_USER"] = tc["user"]

    model = ModelGiaMayBay(kich_ban=tc_id, kieu="hybrid")
    tools = tools_langchain()
    harness = FlightHarnessMiddleware()
    middleware = [
        harness,
        ModelCallLimitMiddleware(run_limit=14, exit_behavior="end"),
    ]

    agent = create_agent(model=model, tools=tools, middleware=middleware)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Hybrid · {tc_id}: {tc['name']}")
        print(f"{'='*70}")

    start = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": tc["question"]}]},
            {"recursion_limit": 30})
    except Exception as e:
        elapsed = time.time() - start
        if verbose:
            print(f"  LỖI: {type(e).__name__}: {str(e)[:100]}")
        return _make_result(tc_id, "hybrid", [], elapsed, error=str(e))

    elapsed = time.time() - start
    messages = result["messages"]
    if verbose:
        _print_trace(messages)

    return _make_result(tc_id, "hybrid", messages, elapsed)


# ========================================== ĐÁNH GIÁ
def _make_result(tc_id: str, agent_type: str, messages: list,
                 elapsed: float, error: str = "") -> dict:
    """Tạo kết quả đánh giá cho một lần chạy."""
    tc = TEST_CASES[tc_id]

    # Đếm tool calls
    tool_calls = sum(1 for m in messages
                     if getattr(m, "tool_calls", None))

    # Kiểm tra hoàn thành
    completion = CompletionChecker.check_from_messages(messages)

    # Kiểm tra câu trả lời cuối
    final_answer = ""
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai" and str(getattr(m, "content", "")).strip():
            final_answer = str(m.content)
            break

    # Kiểm tra dừng đúng
    has_handoff = "DỪNG BẤT THƯỜNG" in final_answer
    stopped_correctly = True
    if tc["expect_success"] and has_handoff:
        stopped_correctly = False  # Không nên dừng bất thường khi expected success
    if not tc["expect_success"] and not has_handoff and not ("không" in final_answer.lower()
            or "liên hệ" in final_answer.lower()):
        stopped_correctly = False

    # Kiểm căn cứ
    tool_results = [str(m.content) for m in messages if getattr(m, "type", None) == "tool"]
    if has_handoff:
        fact_check_pass = True  # Báo cáo dừng bất thường / từ chối quyền, không phải bịa dữ kiện
    else:
        fact_check = FlightFactChecker.check(final_answer, tool_results)
        fact_check_pass = fact_check["dat"]

    # Có bàn giao đủ thông tin không
    handoff_complete = True
    if has_handoff:
        handoff_complete = all(k in final_answer for k in
                               ["Đã thử", "Trạng thái", "Hỏi người"])

    # Có plan steps? (cho Plan và Hybrid)
    plan_steps = sum(1 for m in messages
                     if getattr(m, "type", None) == "ai"
                     and "KẾ HOẠCH" in str(getattr(m, "content", "")))
    replan_steps = sum(1 for m in messages
                       if getattr(m, "type", None) == "ai"
                       and "ĐIỀU CHỈNH" in str(getattr(m, "content", "")))

    return {
        "tc_id": tc_id,
        "tc_name": tc["name"],
        "agent": agent_type,
        "tool_calls": tool_calls,
        "plan_steps": plan_steps,
        "replan_steps": replan_steps,
        "total_steps": len(messages) - 1,   # trừ human message
        "elapsed": round(elapsed, 3),
        "stopped_correctly": stopped_correctly,
        "booking_complete": completion.get("complete", False),
        "fact_check_pass": fact_check_pass,
        "handoff_complete": handoff_complete,
        "error": error,
        "final_answer_preview": final_answer[:120],
    }


def _print_trace(messages: list) -> None:
    """In trace phiên làm việc của agent."""
    step = 0
    for m in messages:
        if getattr(m, "type", None) == "human":
            print(f'  [V1] Human  "{str(m.content)[:80]}…"')
        elif getattr(m, "type", None) == "ai":
            content = str(getattr(m, "content", "")).strip()
            tool_calls = getattr(m, "tool_calls", None)
            if content and not tool_calls:
                for line in content.splitlines():
                    print(f"  [V{step}] AI     {line}")
            elif content and tool_calls:
                for line in content.splitlines():
                    print(f"  [V{step}] AI     {line}")
                step += 1
                for c in tool_calls:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                    print(f"  [V{step}] AI     {c['name']}({args_str})")
            elif tool_calls:
                step += 1
                for c in tool_calls:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                    print(f"  [V{step}] AI     {c['name']}({args_str})")
        elif getattr(m, "type", None) == "tool":
            content = str(m.content)
            if len(content) > 80:
                content = content[:77] + "..."
            print(f"  [V{step}] Tool   {content}")


def run_all(verbose: bool = True) -> list[dict]:
    """Chạy tất cả test cases cho cả 3 mẫu agent, trả danh sách kết quả."""
    runners = {
        "react": run_react,
        "plan": run_plan_execute,
        "hybrid": run_hybrid,
    }
    results = []
    for agent_name, runner in runners.items():
        for tc_id in TEST_CASES:
            r = runner(tc_id, verbose=verbose)
            results.append(r)
    return results


def print_evaluation_table(results: list[dict]) -> None:
    """In bảng so sánh đánh giá."""
    print(f"\n{'='*90}")
    print("BẢNG ĐÁNH GIÁ SO SÁNH BA MẪU AGENT")
    print(f"{'='*90}")

    # Header
    print(f"{'TC':<5} {'Agent':<8} {'Tool':>5} {'Plan':>5} {'RePlan':>6} "
          f"{'Dừng đúng':>10} {'Booking':>8} {'Căn cứ':>7} {'Bàn giao':>9}")
    print("-" * 90)

    for r in results:
        dung = "✓" if r["stopped_correctly"] else "✗"
        booking = "✓" if r["booking_complete"] else "—"
        fact = "✓" if r["fact_check_pass"] else "✗"
        handoff = "✓" if r["handoff_complete"] else "—"

        print(f"{r['tc_id']:<5} {r['agent']:<8} {r['tool_calls']:>5} "
              f"{r['plan_steps']:>5} {r['replan_steps']:>6} "
              f"{dung:>10} {booking:>8} {fact:>7} {handoff:>9}")

    # Tổng hợp theo agent
    print(f"\n{'─'*60}")
    print("TỔNG HỢP THEO MẪU AGENT:")
    print(f"{'─'*60}")
    for agent in ("react", "plan", "hybrid"):
        agent_results = [r for r in results if r["agent"] == agent]
        avg_tools = sum(r["tool_calls"] for r in agent_results) / len(agent_results)
        correct = sum(1 for r in agent_results if r["stopped_correctly"])
        fact_ok = sum(1 for r in agent_results if r["fact_check_pass"])
        n = len(agent_results)
        print(f"  {agent:<8}  TB tool calls: {avg_tools:.1f}  "
              f"Dừng đúng: {correct}/{n}  Căn cứ OK: {fact_ok}/{n}")


# ========================================== MAIN
def main() -> None:
    p = argparse.ArgumentParser(description="BTVN#3 · Agent Đặt Vé Máy Bay")
    p.add_argument("--agent", choices=["react", "plan", "hybrid"],
                   help="Chọn mẫu agent để chạy")
    p.add_argument("--tc", type=int, choices=[1, 2, 3, 4, 5],
                   help="Chỉ chạy test case cụ thể")
    p.add_argument("--all", action="store_true",
                   help="Chạy tất cả + bảng đánh giá")
    p.add_argument("--quiet", action="store_true",
                   help="Chỉ in bảng đánh giá, không in trace")
    args = p.parse_args()

    if args.all:
        results = run_all(verbose=not args.quiet)
        print_evaluation_table(results)
        return

    runners = {
        "react": run_react,
        "plan": run_plan_execute,
        "hybrid": run_hybrid,
    }

    agents = [args.agent] if args.agent else ["react", "plan", "hybrid"]
    tcs = [f"tc{args.tc}"] if args.tc else list(TEST_CASES.keys())

    results = []
    for agent_name in agents:
        runner = runners[agent_name]
        for tc_id in tcs:
            r = runner(tc_id, verbose=not args.quiet)
            results.append(r)

    if len(results) > 1:
        print_evaluation_table(results)


if __name__ == "__main__":
    sys.exit(main())
