# -*- coding: utf-8 -*-
"""BTVN#3 · Các lớp harness cho Agent Đặt Vé Máy Bay.

Sáu lớp harness, Python thuần, không gọi model, không phụ thuộc LangChain:

    DataConstraintChecker    kiểm ràng buộc dữ liệu đầu vào tool
    CompletionChecker        kiểm tiêu chí hoàn thành nhiệm vụ
    PermissionChecker        kiểm quyền người dùng
    HandoffReporter          sinh báo cáo bàn giao khi dừng bất thường
    FlightLoopDetector       phát hiện lặp (kế thừa LoopDetector demo)
    FlightFactChecker        kiểm căn cứ câu trả lời (kế thừa kiem_can_cu demo)

Chạy thử:  python lib/harness_maybay.py
"""
from __future__ import annotations

import os
import sys
import re
from collections import deque
from typing import Any

# Cho phép chạy standalone: python lib/harness_maybay.py
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.tools_maybay import SAN_BAY, CHUYEN_BAY, NGUOI_DUNG, DAT_VE


# ================================================ 1 · RÀNG BUỘC DỮ LIỆU
class DataConstraintChecker:
    """Kiểm tra ràng buộc dữ liệu trước khi tool thực thi.

    Dùng để chặn lời gọi tool không hợp lệ ngay từ đầu, tiết kiệm
    vòng lặp agent.
    """

    VALID_AIRPORTS = set(SAN_BAY.keys())
    VALID_SEAT_CLASSES = {"economy", "business"}
    VALID_PAYMENTS = {"credit_card", "bank_transfer", "ewallet"}
    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    @classmethod
    def check_airport(cls, code: str) -> str | None:
        """Trả thông báo lỗi nếu mã sân bay không hợp lệ, None nếu OK."""
        if code.upper().strip() not in cls.VALID_AIRPORTS:
            return (f"Mã sân bay '{code}' không hợp lệ. "
                    f"Mã hợp lệ: {', '.join(sorted(cls.VALID_AIRPORTS))}")
        return None

    @classmethod
    def check_date(cls, date: str) -> str | None:
        if not cls.DATE_PATTERN.match(date.strip()):
            return f"Ngày '{date}' không đúng định dạng YYYY-MM-DD."
        return None

    @classmethod
    def check_seat_class(cls, sc: str) -> str | None:
        if sc.lower().strip() not in cls.VALID_SEAT_CLASSES:
            return f"Hạng ghế '{sc}' không hợp lệ. Chọn: economy hoặc business."
        return None

    @classmethod
    def check_passenger_name(cls, name: str) -> str | None:
        if not name or not name.strip():
            return "Tên hành khách không được để trống."
        if len(name.strip()) < 2:
            return "Tên hành khách phải có ít nhất 2 ký tự."
        return None

    @classmethod
    def check_payment(cls, pm: str) -> str | None:
        if pm.lower().strip() not in cls.VALID_PAYMENTS:
            return (f"Phương thức thanh toán '{pm}' không hợp lệ. "
                    f"Chọn: credit_card, bank_transfer, ewallet.")
        return None

    @classmethod
    def validate_search(cls, departure: str, destination: str, date: str) -> list[str]:
        errors = []
        e = cls.check_airport(departure)
        if e: errors.append(e)
        e = cls.check_airport(destination)
        if e: errors.append(e)
        e = cls.check_date(date)
        if e: errors.append(e)
        if departure.upper() == destination.upper():
            errors.append("Điểm đi và điểm đến không được trùng nhau.")
        return errors

    @classmethod
    def validate_booking(cls, flight_id: str, passenger: str,
                         seat_class: str, payment: str) -> list[str]:
        errors = []
        if flight_id.upper().strip() not in CHUYEN_BAY:
            errors.append(f"Chuyến bay '{flight_id}' không tồn tại.")
        e = cls.check_passenger_name(passenger)
        if e: errors.append(e)
        e = cls.check_seat_class(seat_class)
        if e: errors.append(e)
        e = cls.check_payment(payment)
        if e: errors.append(e)
        return errors


# ============================================ 2 · TIÊU CHÍ HOÀN THÀNH
class CompletionChecker:
    """Kiểm tra nhiệm vụ đặt vé đã hoàn thành hay chưa.

    Tiêu chí:
    - Có booking_id hợp lệ
    - Status = confirmed
    - Đã thu thập đủ thông tin bắt buộc
    """

    REQUIRED_FIELDS = {"flight_id", "passenger", "seat_class", "payment_method"}

    @classmethod
    def is_booking_complete(cls, booking_state: dict) -> dict:
        """Kiểm tra trạng thái booking.

        Returns:
            dict với keys: complete (bool), missing (list), reason (str)
        """
        if not booking_state:
            return {"complete": False, "missing": list(cls.REQUIRED_FIELDS),
                    "reason": "Chưa có thông tin booking nào."}

        missing = []
        for field in cls.REQUIRED_FIELDS:
            if field not in booking_state or not booking_state[field]:
                missing.append(field)

        if missing:
            return {"complete": False, "missing": missing,
                    "reason": f"Thiếu thông tin: {', '.join(missing)}"}

        bk_id = booking_state.get("booking_id")
        if not bk_id:
            return {"complete": False, "missing": ["booking_id"],
                    "reason": "Chưa có mã đặt chỗ. Cần gọi book_flight."}

        status = booking_state.get("status", "")
        if status != "confirmed":
            return {"complete": False, "missing": [],
                    "reason": f"Booking chưa xác nhận. Trạng thái: {status}"}

        return {"complete": True, "missing": [],
                "reason": f"Đặt vé hoàn thành. Mã: {bk_id}, trạng thái: confirmed"}

    @classmethod
    def extract_booking_from_messages(cls, messages: list) -> dict:
        """Trích xuất trạng thái booking từ lịch sử message."""
        state = {}
        for m in messages:
            if getattr(m, "type", None) != "tool":
                continue
            content = m.content if isinstance(m.content, str) else str(m.content)
            try:
                import json
                data = json.loads(content)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("status") == "ok":
                if "booking_id" in data:
                    state.update({
                        "booking_id": data.get("booking_id"),
                        "flight_id": data.get("flight_id"),
                        "passenger": data.get("passenger"),
                        "seat_class": data.get("seat_class"),
                        "payment_method": data.get("payment"),
                        "status": data.get("booking_status", ""),
                    })
        return state

    @classmethod
    def check_from_messages(cls, messages: list) -> dict:
        state = cls.extract_booking_from_messages(messages)
        return cls.is_booking_complete(state)


# ================================================== 3 · KIỂM QUYỀN
class PermissionChecker:
    """Kiểm tra quyền người dùng trước khi thực hiện hành động.

    Ba quy tắc:
    1. Chỉ chủ booking mới được huỷ
    2. Chỉ VIP mới đặt business
    3. Giới hạn số vé mỗi ngày
    """

    @staticmethod
    def can_cancel(user_id: str, booking_id: str) -> dict:
        bk = DAT_VE.get(booking_id.upper().strip())
        if not bk:
            return {"allowed": False, "reason": f"Booking '{booking_id}' không tồn tại."}
        if bk["owner"] != user_id:
            return {"allowed": False,
                    "reason": f"Người dùng '{user_id}' không phải chủ booking '{booking_id}'.",
                    "owner": bk["owner"]}
        return {"allowed": True, "reason": "OK"}

    @staticmethod
    def can_book_business(user_id: str) -> dict:
        user = NGUOI_DUNG.get(user_id)
        if not user:
            return {"allowed": False, "reason": f"Người dùng '{user_id}' không tồn tại."}
        if not user["vip"]:
            return {"allowed": False,
                    "reason": "Chỉ thành viên VIP mới đặt hạng business.",
                    "hint": "Đặt hạng economy hoặc nâng cấp VIP."}
        return {"allowed": True, "reason": "OK"}

    @staticmethod
    def can_book_today(user_id: str) -> dict:
        user = NGUOI_DUNG.get(user_id)
        if not user:
            return {"allowed": False, "reason": f"Người dùng '{user_id}' không tồn tại."}
        if user["bookings_today"] >= user["max_per_day"]:
            return {"allowed": False,
                    "reason": f"Đã đạt giới hạn {user['max_per_day']} vé/ngày.",
                    "hint": "Thử lại vào ngày mai."}
        return {"allowed": True, "reason": "OK",
                "remaining": user["max_per_day"] - user["bookings_today"]}

    @classmethod
    def check_all(cls, user_id: str, action: str, **kwargs) -> dict:
        """Kiểm tra tổng hợp quyền cho một hành động."""
        if action == "cancel":
            return cls.can_cancel(user_id, kwargs.get("booking_id", ""))
        elif action == "book":
            # Kiểm tra giới hạn ngày
            day_check = cls.can_book_today(user_id)
            if not day_check["allowed"]:
                return day_check
            # Kiểm tra hạng business
            if kwargs.get("seat_class", "").lower() == "business":
                biz_check = cls.can_book_business(user_id)
                if not biz_check["allowed"]:
                    return biz_check
            return {"allowed": True, "reason": "OK"}
        return {"allowed": True, "reason": "OK"}


# ================================================== 4 · BÀN GIAO
class HandoffReporter:
    """Sinh báo cáo bàn giao khi agent dừng bất thường.

    Bốn trường bắt buộc (theo slide S50):
    1. stop_reason    — vì sao dừng
    2. actions_tried  — chuỗi hành động đã thử
    3. booking_state  — trạng thái hiện tại
    4. question       — câu hỏi cho người xử lý tiếp
    """

    @staticmethod
    def create(stop_reason: str, actions_tried: list,
               booking_state: dict, question: str) -> dict:
        return {
            "stop_reason": stop_reason,
            "actions_tried": actions_tried,
            "booking_state": booking_state,
            "question_for_human": question,
        }

    @staticmethod
    def format(report: dict) -> str:
        actions = " → ".join(report.get("actions_tried", []))
        state = str(report.get("booking_state", {}))
        return (
            "DỪNG BẤT THƯỜNG · " + report["stop_reason"] + "\n"
            "  Đã thử          : " + (actions or "(chưa thử gì)") + "\n"
            "  Trạng thái      : " + state + "\n"
            "  Hỏi người dùng  : " + report["question_for_human"]
        )


# ========================================= 5 · PHÁT HIỆN LẶP CHUYẾN BAY
class FlightLoopDetector:
    """Phát hiện lặp trong vòng lặp agent đặt vé.

    Kế thừa logic LoopDetector từ demo:
    1. Trùng action: cùng (tool, args) lặp lại
    2. Trùng observation: kết quả giống hệt nhau
    3. Không tiến triển: trạng thái đứng yên

    Mở rộng: phân biệt retry hợp lệ (polling get_booking_status)
    với lặp vô ích.
    """

    POLLING_TOOLS = {"get_booking_status"}  # retry hợp lệ

    def __init__(self, window: int = 6, repeat_k: int = 2,
                 same_obs_k: int = 4, stall_n: int = 5):
        self.recent: deque = deque(maxlen=window)
        self.obs: deque = deque(maxlen=window)
        self.k = repeat_k
        self.k_obs = same_obs_k
        self.n = stall_n
        self.last_progress: Any = None
        self.stall = 0

    def check(self, tool: str, args: dict,
              observation: Any = None, progress: Any = None) -> str | None:
        """Trả chuỗi mô tả nếu phát hiện lặp, None nếu bình thường."""

        # Bỏ qua polling tools (retry hợp lệ)
        if tool in self.POLLING_TOOLS:
            return None

        # Tín hiệu 1: trùng action
        fp = (tool, repr(sorted(args.items())))
        if self.recent.count(fp) + 1 >= self.k:
            return (f"LOOP · '{tool}' gọi {self.recent.count(fp) + 1} lần "
                    f"với cùng tham số trong {self.recent.maxlen} vòng gần nhất")
        self.recent.append(fp)

        # Tín hiệu 2: trùng observation
        if observation is not None:
            ofp = repr(observation)
            if self.obs.count(ofp) + 1 >= self.k_obs:
                return (f"LOOP · {self.obs.count(ofp) + 1} lời gọi khác tham số "
                        f"nhưng trả về cùng một kết quả")
            self.obs.append(ofp)

        # Tín hiệu 3: không tiến triển
        if progress is not None:
            self.stall = self.stall + 1 if progress == self.last_progress else 0
            self.last_progress = progress
            if self.stall >= self.n:
                return f"STALL · tiến triển đứng yên ở {progress!r} qua {self.stall} vòng"

        return None


# ============================================ 6 · KIỂM CĂN CỨ CHUYẾN BAY
class FlightFactChecker:
    """Kiểm tra câu trả lời của agent có căn cứ từ kết quả tool hay không.

    Kế thừa logic kiem_can_cu từ demo, mở rộng mẫu regex cho đặt vé.
    """

    PATTERNS = [
        r"\d{1,3}(?:\.\d{3})+(?:đ|\s?VNĐ|\s?đồng)?",    # 1.500.000đ
        r"\d{1,2}:\d{2}",                                  # 06:00
        r"\d{4}-\d{2}-\d{2}",                              # 2024-12-20
        r"[A-Z]{2}\d{3}",                                  # VN100, VJ200
        r"BK\d{3}",                                         # BK001
        r"(?:HAN|SGN|DAD|CXR|PQC|HUI)",                   # mã sân bay
    ]

    @classmethod
    def extract_facts(cls, text: str) -> list[str]:
        """Trích xuất mọi dữ kiện (số, mã, ngày giờ) từ câu trả lời."""
        facts = []
        for pattern in cls.PATTERNS:
            facts += re.findall(pattern, text)
        return list(dict.fromkeys(facts))

    @classmethod
    def check(cls, answer: str, tool_results: list[str]) -> dict:
        """Đối chiếu dữ kiện trong câu trả lời với kết quả tool.

        Args:
            answer:       câu trả lời cuối cùng của agent
            tool_results: danh sách nội dung các ToolMessage

        Returns:
            dict: dat (bool), chi_tiet (list), khong_co_nguon (list)
        """
        source = " ".join(str(x) for x in tool_results)
        details = [{"fact": f, "has_source": f in source}
                   for f in cls.extract_facts(answer)]
        missing = [d["fact"] for d in details if not d["has_source"]]
        return {"dat": not missing, "chi_tiet": details, "khong_co_nguon": missing}

    @classmethod
    def safe_answer(cls, answer: str, tool_results: list[str]) -> str:
        """Trả câu trả lời an toàn. Chặn nếu có dữ kiện bịa."""
        result = cls.check(answer, tool_results)
        if result["dat"]:
            return answer
        return ("Chưa trả lời được đầy đủ. Các dữ kiện sau không truy được "
                "về kết quả tool nào: "
                + " · ".join(result["khong_co_nguon"])
                + ". Cần bổ sung tool hoặc chuyển cho nhân viên hỗ trợ.")


# ============================================================ SELF-TEST
if __name__ == "__main__":
    print("── DataConstraintChecker ──")
    print("  Sân bay XYZ:", DataConstraintChecker.check_airport("XYZ"))
    print("  Sân bay HAN:", DataConstraintChecker.check_airport("HAN"))
    print("  Ngày sai:", DataConstraintChecker.check_date("20-12-2024"))
    print("  Ngày đúng:", DataConstraintChecker.check_date("2024-12-20"))
    print("  Validate search:",
          DataConstraintChecker.validate_search("HAN", "SGN", "2024-12-20"))
    print("  Validate search lỗi:",
          DataConstraintChecker.validate_search("XYZ", "SGN", "bad-date"))

    print("\n── CompletionChecker ──")
    print("  Rỗng:", CompletionChecker.is_booking_complete({}))
    print("  Thiếu:", CompletionChecker.is_booking_complete(
        {"flight_id": "VN100", "passenger": "A"}))
    print("  Đủ:", CompletionChecker.is_booking_complete(
        {"flight_id": "VN100", "passenger": "A", "seat_class": "economy",
         "payment_method": "credit_card", "booking_id": "BK003",
         "status": "confirmed"}))

    print("\n── PermissionChecker ──")
    print("  user_a cancel BK001:", PermissionChecker.can_cancel("user_a", "BK001"))
    print("  user_a cancel BK002:", PermissionChecker.can_cancel("user_a", "BK002"))
    print("  user_a book business:", PermissionChecker.can_book_business("user_a"))
    print("  user_b book business:", PermissionChecker.can_book_business("user_b"))
    print("  user_c book today:", PermissionChecker.can_book_today("user_c"))

    print("\n── FlightLoopDetector ──")
    det = FlightLoopDetector()
    for i, fid in enumerate(["VN100", "VN101", "VN100"], 1):
        w = det.check("search_flights",
                       {"departure": "HAN", "destination": "SGN", "date": "2024-12-20"} if fid == "VN100"
                       else {"departure": "HAN", "destination": "SGN", "date": "2024-12-21"},
                       progress=0)
        print(f"  V{i} search_flights → {w or 'bình thường'}")

    print("\n── FlightFactChecker ──")
    answer = "Chuyến VN100 khởi hành 06:00, giá 1.500.000đ, mã BK003"
    tools = ['{"flight_id":"VN100","time_depart":"06:00","price":"1.500.000đ"}']
    result = FlightFactChecker.check(answer, tools)
    for d in result["chi_tiet"]:
        tag = "có nguồn" if d["has_source"] else "BỊA"
        print(f"  {tag:>8} {d['fact']}")

    print("\n── HandoffReporter ──")
    report = HandoffReporter.create(
        stop_reason="Hết ghế trên chuyến yêu cầu",
        actions_tried=["search_flights(HAN→SGN)", "check_seat(VN101)"],
        booking_state={"flight_id": "VN101", "status": "no_seats"},
        question="Chuyển sang chuyến VN100 hay đổi ngày?",
    )
    print(HandoffReporter.format(report))
