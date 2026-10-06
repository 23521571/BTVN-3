# -*- coding: utf-8 -*-
"""BTVN#3 · Bộ tool đặt vé máy bay (mockup).

Dữ liệu tĩnh, không gọi mạng. Sáu tool mô phỏng hệ thống đặt vé:

    search_flights(departure, destination, date)          tìm chuyến bay
    get_flight_details(flight_id)                         chi tiết chuyến bay
    check_seat_availability(flight_id, seat_class)        kiểm tra ghế trống
    book_flight(flight_id, passenger, seat_class, payment) đặt vé
    cancel_booking(booking_id)                            huỷ vé
    get_booking_status(booking_id)                        tra cứu booking
"""
import copy
import os

# ================================================================ dữ liệu
SAN_BAY = {
    "HAN": "Nội Bài (Hà Nội)",
    "SGN": "Tân Sơn Nhất (TP.HCM)",
    "DAD": "Đà Nẵng",
    "CXR": "Cam Ranh (Nha Trang)",
    "PQC": "Phú Quốc",
    "HUI": "Phú Bài (Huế)",
}

CHUYEN_BAY = {
    "VN100": dict(flight_id="VN100", departure="HAN", destination="SGN",
                  date="2024-12-20", airline="Vietnam Airlines",
                  time_depart="06:00", time_arrive="08:10",
                  price_economy=1_500_000, price_business=3_500_000,
                  seats_economy=5, seats_business=2),
    "VN101": dict(flight_id="VN101", departure="HAN", destination="SGN",
                  date="2024-12-20", airline="Vietnam Airlines",
                  time_depart="14:30", time_arrive="16:40",
                  price_economy=1_800_000, price_business=3_800_000,
                  seats_economy=0, seats_business=1),      # hết economy
    "VJ200": dict(flight_id="VJ200", departure="SGN", destination="DAD",
                  date="2024-12-21", airline="VietJet Air",
                  time_depart="08:15", time_arrive="09:35",
                  price_economy=1_200_000, price_business=2_800_000,
                  seats_economy=10, seats_business=3),
    "VJ201": dict(flight_id="VJ201", departure="SGN", destination="DAD",
                  date="2024-12-21", airline="VietJet Air",
                  time_depart="16:45", time_arrive="18:05",
                  price_economy=1_350_000, price_business=2_900_000,
                  seats_economy=8, seats_business=0),       # hết business
    "QH300": dict(flight_id="QH300", departure="HAN", destination="DAD",
                  date="2024-12-22", airline="Bamboo Airways",
                  time_depart="10:00", time_arrive="11:20",
                  price_economy=1_400_000, price_business=3_200_000,
                  seats_economy=3, seats_business=2),
}

# Booking hiện có (để test cancel + get_status)
DAT_VE = {
    "BK001": dict(booking_id="BK001", flight_id="VN100",
                  passenger="Nguyen Van A", seat_class="economy",
                  status="confirmed", owner="user_a",
                  price=1_500_000),
    "BK002": dict(booking_id="BK002", flight_id="VJ200",
                  passenger="Tran Thi B", seat_class="business",
                  status="confirmed", owner="user_b",
                  price=2_800_000),
}

NGUOI_DUNG = {
    "user_a": dict(name="Nguyen Van A", vip=False, bookings_today=0, max_per_day=3),
    "user_b": dict(name="Tran Thi B", vip=True, bookings_today=2, max_per_day=5),
    "user_c": dict(name="Le Van C", vip=False, bookings_today=3, max_per_day=3),
}

# Bộ đếm booking mới
_next_bk = [3]  # BK003, BK004, ...


def _current_user() -> str:
    return os.environ.get("BTVN3_USER", "user_a")


# Bản sao ban đầu của CHUYEN_BAY để reset
_BAN_SAO_CHUYEN_BAY = copy.deepcopy(CHUYEN_BAY)


def reset_data():
    """Reset dữ liệu về trạng thái ban đầu (cho chạy test lặp lại)."""
    global DAT_VE, NGUOI_DUNG, CHUYEN_BAY
    _next_bk[0] = 3
    CHUYEN_BAY.clear()
    CHUYEN_BAY.update(copy.deepcopy(_BAN_SAO_CHUYEN_BAY))
    DAT_VE = {
        "BK001": dict(booking_id="BK001", flight_id="VN100",
                      passenger="Nguyen Van A", seat_class="economy",
                      status="confirmed", owner="user_a", price=1_500_000),
        "BK002": dict(booking_id="BK002", flight_id="VJ200",
                      passenger="Tran Thi B", seat_class="business",
                      status="confirmed", owner="user_b", price=2_800_000),
    }
    NGUOI_DUNG.update({
        "user_a": dict(name="Nguyen Van A", vip=False, bookings_today=0, max_per_day=3),
        "user_b": dict(name="Tran Thi B", vip=True, bookings_today=2, max_per_day=5),
        "user_c": dict(name="Le Van C", vip=False, bookings_today=3, max_per_day=3),
    })


def _gia_vn(amount: int) -> str:
    """Định dạng số tiền kiểu Việt Nam với dấu chấm phân cách ngàn."""
    return f"{amount:,}".replace(",", ".") + "đ"


# ================================================================ tool 1
def search_flights(departure: str, destination: str, date: str) -> dict:
    """Tìm chuyến bay theo điểm đi, điểm đến và ngày.

    Tham số:
        departure:    mã sân bay đi (VD: HAN, SGN, DAD)
        destination:  mã sân bay đến
        date:         ngày bay (YYYY-MM-DD)
    """
    dep = departure.upper().strip()
    dest = destination.upper().strip()

    if dep not in SAN_BAY:
        return {"status": "error", "error": "invalid_departure",
                "message": f"Mã sân bay '{departure}' không hợp lệ",
                "valid_airports": list(SAN_BAY.keys())}
    if dest not in SAN_BAY:
        return {"status": "error", "error": "invalid_destination",
                "message": f"Mã sân bay '{destination}' không hợp lệ",
                "valid_airports": list(SAN_BAY.keys())}

    hits = [f for f in CHUYEN_BAY.values()
            if f["departure"] == dep and f["destination"] == dest and f["date"] == date]
    if not hits:
        return {"status": "ok", "departure": dep, "destination": dest, "date": date,
                "matched": 0, "results": [],
                "hint": "Không có chuyến bay phù hợp. Thử đổi ngày hoặc tuyến bay khác."}

    results = [{"flight_id": f["flight_id"], "airline": f["airline"],
                "departure": dep, "destination": dest, "date": date,
                "time": f"{f['time_depart']}-{f['time_arrive']}",
                "price_economy": _gia_vn(f['price_economy']),
                "price_business": _gia_vn(f['price_business'])}
               for f in hits]
    return {"status": "ok", "departure": dep, "destination": dest, "date": date,
            "matched": len(results), "results": results}


# ================================================================ tool 2
def get_flight_details(flight_id: str) -> dict:
    """Xem chi tiết một chuyến bay theo mã chuyến.

    Tham số:
        flight_id:  mã chuyến bay (VD: VN100, VJ200)
    """
    f = CHUYEN_BAY.get(flight_id.upper().strip())
    if not f:
        return {"status": "error", "error": "flight_not_found",
                "message": f"Không tìm thấy chuyến bay '{flight_id}'"}
    info = copy.deepcopy(f)
    info["departure_name"] = SAN_BAY.get(f["departure"], f["departure"])
    info["destination_name"] = SAN_BAY.get(f["destination"], f["destination"])
    info["price_economy"] = _gia_vn(f['price_economy'])
    info["price_business"] = _gia_vn(f['price_business'])
    return {"status": "ok", **info}


# ================================================================ tool 3
def check_seat_availability(flight_id: str, seat_class: str) -> dict:
    """Kiểm tra ghế trống của một chuyến bay.

    Tham số:
        flight_id:   mã chuyến bay
        seat_class:  hạng ghế ('economy' hoặc 'business')
    """
    f = CHUYEN_BAY.get(flight_id.upper().strip())
    if not f:
        return {"status": "error", "error": "flight_not_found",
                "message": f"Không tìm thấy chuyến bay '{flight_id}'"}

    sc = seat_class.lower().strip()
    if sc not in ("economy", "business"):
        return {"status": "error", "error": "invalid_seat_class",
                "message": "Hạng ghế phải là 'economy' hoặc 'business'"}

    key = f"seats_{sc}"
    available = f[key]
    price_key = f"price_{sc}"
    return {"status": "ok", "flight_id": f["flight_id"],
            "seat_class": sc, "available": available,
            "price": _gia_vn(f[price_key]),
            "hint": "Hết ghế. Thử chuyến bay khác hoặc hạng ghế khác." if available == 0 else ""}


# ================================================================ tool 4
def book_flight(flight_id: str, passenger_name: str,
                seat_class: str, payment_method: str) -> dict:
    """Đặt vé máy bay.

    Tham số:
        flight_id:       mã chuyến bay
        passenger_name:  tên hành khách
        seat_class:      hạng ghế ('economy' hoặc 'business')
        payment_method:  phương thức thanh toán ('credit_card', 'bank_transfer', 'ewallet')
    """
    f = CHUYEN_BAY.get(flight_id.upper().strip())
    if not f:
        return {"status": "error", "error": "flight_not_found",
                "message": f"Không tìm thấy chuyến bay '{flight_id}'"}

    sc = seat_class.lower().strip()
    if sc not in ("economy", "business"):
        return {"status": "error", "error": "invalid_seat_class",
                "message": "Hạng ghế phải là 'economy' hoặc 'business'"}

    if not passenger_name or not passenger_name.strip():
        return {"status": "error", "error": "missing_passenger",
                "message": "Tên hành khách không được để trống"}

    pm = payment_method.lower().strip()
    if pm not in ("credit_card", "bank_transfer", "ewallet"):
        return {"status": "error", "error": "invalid_payment",
                "message": "Phương thức thanh toán phải là 'credit_card', 'bank_transfer' hoặc 'ewallet'"}

    # Kiểm tra ghế trống
    key = f"seats_{sc}"
    if f[key] <= 0:
        return {"status": "error", "error": "no_seats",
                "message": f"Hết ghế hạng {sc} trên chuyến {flight_id}",
                "hint": "Thử chuyến bay khác hoặc hạng ghế khác."}

    # Kiểm tra quyền đặt business
    uid = _current_user()
    user = NGUOI_DUNG.get(uid)
    if sc == "business" and user and not user["vip"]:
        return {"status": "error", "error": "permission_denied",
                "message": "Chỉ thành viên VIP mới được đặt hạng business",
                "hint": "Đặt hạng economy hoặc nâng cấp tài khoản VIP."}

    # Kiểm tra giới hạn booking/ngày
    if user and user["bookings_today"] >= user["max_per_day"]:
        return {"status": "error", "error": "daily_limit",
                "message": f"Đã đạt giới hạn {user['max_per_day']} vé/ngày",
                "hint": "Thử lại vào ngày mai hoặc liên hệ hỗ trợ."}

    # Đặt vé thành công
    price_key = f"price_{sc}"
    bk_id = f"BK{_next_bk[0]:03d}"
    _next_bk[0] += 1
    f[key] -= 1

    booking = dict(booking_id=bk_id, flight_id=flight_id,
                   passenger=passenger_name.strip(), seat_class=sc,
                   status="confirmed", owner=uid,
                   price=f[price_key],
                   payment=pm)
    DAT_VE[bk_id] = booking

    if user:
        user["bookings_today"] += 1

    return {"status": "ok", "booking_id": bk_id,
            "flight_id": flight_id, "passenger": passenger_name.strip(),
            "seat_class": sc, "price": _gia_vn(f[price_key]),
            "payment": pm, "booking_status": "confirmed",
            "message": f"Đặt vé thành công! Mã đặt chỗ: {bk_id}"}


# ================================================================ tool 5
def cancel_booking(booking_id: str) -> dict:
    """Huỷ một booking đã đặt.

    Tham số:
        booking_id:  mã đặt chỗ (VD: BK001)
    """
    bk = DAT_VE.get(booking_id.upper().strip())
    if not bk:
        return {"status": "error", "error": "booking_not_found",
                "message": f"Không tìm thấy booking '{booking_id}'"}

    uid = _current_user()
    if bk["owner"] != uid:
        return {"status": "error", "error": "permission_denied",
                "message": f"Bạn không có quyền huỷ booking '{booking_id}'. "
                           f"Booking này thuộc về người dùng khác.",
                "hint": "Chỉ chủ booking mới được phép huỷ."}

    if bk["status"] == "cancelled":
        return {"status": "error", "error": "already_cancelled",
                "message": f"Booking '{booking_id}' đã bị huỷ trước đó."}

    bk["status"] = "cancelled"
    # Trả lại ghế
    f = CHUYEN_BAY.get(bk["flight_id"])
    if f:
        f[f"seats_{bk['seat_class']}"] += 1

    return {"status": "ok", "booking_id": booking_id,
            "new_status": "cancelled",
            "refund": _gia_vn(bk['price']),
            "message": f"Đã huỷ booking {booking_id}. Hoàn tiền {_gia_vn(bk['price'])}."}


# ================================================================ tool 6
def get_booking_status(booking_id: str) -> dict:
    """Tra cứu trạng thái một booking.

    Tham số:
        booking_id:  mã đặt chỗ (VD: BK001)
    """
    bk = DAT_VE.get(booking_id.upper().strip())
    if not bk:
        return {"status": "error", "error": "booking_not_found",
                "message": f"Không tìm thấy booking '{booking_id}'"}

    f = CHUYEN_BAY.get(bk["flight_id"], {})
    return {"status": "ok", "booking_id": bk["booking_id"],
            "flight_id": bk["flight_id"],
            "airline": f.get("airline", "N/A"),
            "route": f"{f.get('departure', '?')} → {f.get('destination', '?')}",
            "date": f.get("date", "N/A"),
            "time": f"{f.get('time_depart', '?')}-{f.get('time_arrive', '?')}",
            "passenger": bk["passenger"],
            "seat_class": bk["seat_class"],
            "price": _gia_vn(bk['price']),
            "booking_status": bk["status"]}


# ================================================ đóng gói cho LangChain
def tools_langchain():
    """Trả danh sách tool LangChain cho create_agent."""
    from langchain_core.tools import tool
    return [tool(search_flights), tool(get_flight_details),
            tool(check_seat_availability), tool(book_flight),
            tool(cancel_booking), tool(get_booking_status)]


ALL_TOOLS = {
    "search_flights": search_flights,
    "get_flight_details": get_flight_details,
    "check_seat_availability": check_seat_availability,
    "book_flight": book_flight,
    "cancel_booking": cancel_booking,
    "get_booking_status": get_booking_status,
}


if __name__ == "__main__":
    import json
    def pp(d): print(json.dumps(d, ensure_ascii=False, indent=2))

    print("── search_flights HAN→SGN ──")
    pp(search_flights("HAN", "SGN", "2024-12-20"))

    print("\n── search_flights HAN→PQC (không có) ──")
    pp(search_flights("HAN", "PQC", "2024-12-25"))

    print("\n── check_seat VN101 economy (hết) ──")
    pp(check_seat_availability("VN101", "economy"))

    print("\n── book_flight VN100 ──")
    pp(book_flight("VN100", "Nguyen Van A", "economy", "credit_card"))

    print("\n── cancel_booking BK002 (sai quyền) ──")
    pp(cancel_booking("BK002"))

    print("\n── get_booking_status BK001 ──")
    pp(get_booking_status("BK001"))
