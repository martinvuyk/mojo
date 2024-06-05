# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""`ZoneInfo` module."""

from pathlib import Path, cwd
from utils import Variant
from collections import OptionalReg

from .calendar import PythonCalendar

alias _cal = PythonCalendar


# @value
@register_passable("trivial")
struct Offset:
    """Only supports hour offsets: [0, 15] and minute offsets
    that are: {0, 30, 45}. Offset sign and minute are assumed
    to be equal in DST and STD and DST adds 1 hour to STD hour,
    unless 2 reserved bits are set which means the offset jumps 30
    minutes or 2 hours from its STD time (this was added because
    of literally [one small island](
        https://en.wikipedia.org/wiki/Lord_Howe_Island)
    and an [Antarctica research station](
        https://es.wikipedia.org/wiki/Base_Troll ))."""

    var hour: UInt8
    """Hour: [0, 15]."""
    var minute: UInt8
    """Minute: {0, 30, 45}."""
    var sign: UInt8
    """Sign: {1, -1}. Positive means east of UTC."""
    var buf: UInt8
    """Buffer."""

    fn __init__(inout self, buf: UInt8):
        """Construct an `Offset` from a buffer.

        Args:
            buf: The buffer.
        """
        self.hour = (buf >> 3) & 0b1111
        self.minute = (buf >> 1) & 0b11
        self.sign = buf >> 7 & 0b1
        self.buf = buf

    fn __init__(inout self, values: Tuple[UInt8, UInt8, UInt8], /):
        """Construct an `Offset` from values.

        Args:
            values: Tuple.
        """

        self = Self(values[0], values[1], values[2])

    fn __init__(inout self, values: Tuple[Int, Int, Int], /):
        """Construct an `Offset` from values.

        Args:
            values: Tuple.
        """

        self = Self(values[0], values[1], values[2])

    fn __init__(inout self, hour: UInt8, minute: UInt8, sign: UInt8):
        """Construct an `Offset` from values.

        Args:
            hour: Hour.
            minute: Minute.
            sign: Sign.
        """
        self.hour = hour
        self.minute = minute
        self.sign = sign
        self.buf = (sign << 7) | (hour << 3) | (minute << 1) | 0

    fn __init__(
        inout self,
        iso_tzd_std: String = "+00:00",
        iso_tzd_dst: String = "+00:00",
    ):
        """Construct an `Offset` (8 bits total) for DST start/end.

        Args:
            iso_tzd_std: String with the full ISO8601 TZD (i.e. +00:00).
            iso_tzd_dst: String with the full ISO8601 TZD (i.e. +00:00).
        """
        try:
            var sign = (0 if iso_tzd_std[0] == "+" else 1)

            var std_h: UInt8 = atol(iso_tzd_std[1:2])
            var dst_h: UInt8 = atol(iso_tzd_dst[1:2])

            var std_m: UInt8 = atol(iso_tzd_std[4:6])
            var dst_m: UInt8 = atol(iso_tzd_std[4:6])
            var jumps_2hours: UInt8 = 0
            if std_m - dst_m == 30:  # "Australia/Lord_Howe"
                std_m = 3  # jumps 30 minutes
            elif (dst_h - std_h) ^ 0b10 == 0:  # "Antarctica/Troll"
                jumps_2hours = 1
            else:
                if std_m == 30:
                    std_m = 1
                elif std_m == 45:
                    std_m = 2
            self.hour = std_h
            self.minute = std_m
            self.sign = sign
            self.buf = (sign << 7) | (std_h << 3) | (std_m << 1) | jumps_2hours
        except:
            self.hour = 0
            self.minute = 0
            self.sign = 1
            self.buf = 1 << 7

    @always_inline("nodebug")
    @staticmethod
    fn from_hash(buf: UInt8) -> Self:
        """Get the values from hash.

        Args:
            buf: The hash.

        Returns:
            Self.
        """
        var self = Self()
        var s = (buf >> 7) & 0b1
        self.sign = 1 if s == 0 else -1
        self.hour = (buf >> 3) & 0b1111
        var m = (buf >> 1) & 0b11
        self.minute = 0 if m == 0 else (30 if m == 1 else 45)
        self.buf = buf
        return self

    fn __eq__(self, other: Self) -> Bool:
        """Whether the given Offset is equal to self.

        Args:
            other: The other Offset.

        Returns:
            The result.
        """
        return self.buf == other.buf


# @value
@register_passable("trivial")
struct TzDT:
    """`TzDT` stores the rules for DST start/end."""

    var month: UInt8
    """Month: Month: [1, 12]."""
    var dow: UInt8
    """Dow: Day of week: [0, 6] (monday - sunday)."""
    var eomon: UInt8
    """Eomon: End of month: {0, 1} Whether to count from the
    beginning of the month or the end."""
    var week: UInt8
    """Week: {0, 1} If week=0 -> first week of the month,
    if it's week=1 -> second week. In the case that
    eomon=1, fw=0 -> last week of the month
    and fw=1 -> second to last."""
    var hour: UInt8
    """Hour: {20, 21, 22, 23, 0, 1, 2, 3} Hour at which DST starts/ends."""
    var buf: UInt16
    """Buffer."""

    fn __init__(
        inout self,
        month: UInt8 = 1,
        dow: UInt8 = 0,
        eomon: UInt8 = 0,
        week: UInt8 = 0,
        hour: UInt8 = 0,
    ):
        """Construct a `TzDT` buffer (12 bits total) for DST start/end.

        Args:
            month: Month: [1, 12].
            dow: Day of week: [0, 6] (monday - sunday).
            eomon: End of month: {0, 1} Whether to count from the
                beginning of the month or the end.
            week: {0, 1} If week=0 -> first week of the month,
                if it's week=1 -> second week. In the case that
                eomon=1, fw=0 -> last week of the month
                and fw=1 -> second to last.
            hour: {20, 21, 22, 23, 0, 1, 2, 3} Hour at which DST starts/ends.
        """
        var h: UInt16 = 0
        var i = 0
        for item in List(20, 21, 22, 23, 0, 1, 2, 3):
            if hour == item[]:
                h = i
                break
            i += 1
        var mon = month.cast[DType.uint16]()
        var d = dow.cast[DType.uint16]()
        var eo = eomon.cast[DType.uint16]()
        var w = week.cast[DType.uint16]()

        self.month = month
        self.dow = dow
        self.eomon = eomon
        self.week = week
        self.hour = hour
        self.buf = (mon << 8) | (d << 5) | (eo << 4) | (w << 3) | h

    @always_inline("nodebug")
    @staticmethod
    fn from_hash(buf: UInt16) -> Self:
        """Get the values from hash.

        Args:
            buf: The hash.

        Returns:
            Self.
        """
        var self = Self()
        self.month = (buf >> 8) & 0b1111
        self.dow = (buf >> 5) & 0b111
        self.eomon = (buf >> 4) & 0b1
        self.week = (buf >> 3) & 0b1
        self.hour = buf & 0b111
        self.buf = buf
        return self

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.buf == other.buf


# @value
@register_passable("trivial")
struct ZoneDST:
    """`ZoneDST` stores both start and end dates, and
    the offset for a timezone with DST."""

    var buf: UInt32
    """Buffer."""

    fn __init__(inout self, dst_start: TzDT, dst_end: TzDT, offset: Offset):
        """Construct a `ZoneDST` from values.

        Args:
            dst_start: TzDT.
            dst_end: TzDT.
            offset: Offset.
        """
        self.buf = (
            (dst_start.buf.cast[DType.uint32]() << 20)
            | (dst_end.buf.cast[DType.uint32]() << 12)
            | offset.buf.cast[DType.uint32]()
        )

    fn __init__(inout self, buf: UInt32):
        """Construct a `ZoneDST` from a buffer.

        Args:
            buf: The buffer.
        """
        self.buf = buf

    @always_inline("nodebug")
    fn from_hash(self) -> (TzDT, TzDT, Offset):
        """Get the values from hash.

        Returns:
            - dst_start: TzDT hash (12 bits in a UInt16 buffer).
            - dst_end: TzDT hash (12 bits in a UInt16 buffer).
            - offset: Offset hash (8 bits in a UInt16 buffer).
        """
        return (
            TzDT.from_hash(
                ((self.buf >> 20) & 0b11111111).cast[DType.uint16]()
            ),
            TzDT.from_hash(
                ((self.buf >> 12) & 0b11111111).cast[DType.uint16]()
            ),
            Offset.from_hash((self.buf & 0b111111111111).cast[DType.uint8]()),
        )


@value
struct ZoneInfoFile32(CollectionElement):
    """ZoneInfoFile to store Offset of tz with DST.
    Zoneinfo that lives in a file. Smallest memory footprint
    but only supports 256 timezones (there are ~ 418).
    """

    var _file: Path

    fn __init__(inout self):
        """Construct a `ZoneInfoFile`."""
        try:
            self._file = cwd() / "zoneinfo_dump"
        except:
            self._file = Path(".") / "zoneinfo_dump"

    fn add(inout self, key: StringLiteral, value: ZoneDST):
        """Add a value to the file.

        Args:
            key: The tz_str.
            value: The ZoneDST with the hash.
        """
        try:
            with open(self._file, "wb") as f:
                _ = f.seek(hash(key) * 32)
                # FIXME: this is horrible
                var items = List[UInt8](
                    UInt8(value.buf[0]),
                    UInt8(value.buf[1]),
                    UInt8(value.buf[2]),
                    UInt8(value.buf[3]),
                    0,
                )
                f.write(String(items))
        except:
            # TODO: propper logging
            print("could not save zoneinfo to file")
            pass

    fn get(self, key: StringLiteral) -> OptionalReg[ZoneDST]:
        """Get a value from the file.

        Args:
            key: The tz_str.

        Returns:
            An Optional `ZoneDST`.
        """
        try:
            var value: UInt32
            with open(self._file, "rb") as f:
                _ = f.seek(hash(key) * 32)
                var bufs = f.read_bytes(4)
                value = (
                    (bufs[0].cast[DType.uint32]() << 24)
                    | (bufs[1].cast[DType.uint32]() << 16)
                    | (bufs[2].cast[DType.uint32]() << 8)
                    | bufs[3].cast[DType.uint32]()
                )
            return ZoneDST(value)
        except:
            return None

    fn __del__(owned self):
        """Delete the file."""
        try:
            import os

            os.remove(self._file)
        except:
            # TODO: propper logging
            print("could not delete zoneinfo file")
            pass


@value
struct ZoneInfoFile8(CollectionElement):
    """ZoneInfoFile to store Offset of tz with no DST.
    Zoneinfo that lives in a file. Smallest memory footprint
    but only supports 256 timezones (there are ~ 418).
    """

    var _file: Path

    fn __init__(inout self):
        """Construct a `ZoneInfoFile`."""
        try:
            self._file = cwd() / "zoneinfo_dump"
        except:
            self._file = "./zoneinfo_dump"

    fn add(inout self, key: StringLiteral, value: Offset):
        """Add a value to the file.

        Args:
            key: The tz_str.
            value: The buffer with the hash.
        """
        var index = hash(key)
        try:
            with open(self._file, "wb") as f:
                _ = f.seek(index * 8)
                # FIXME: this is horrible
                f.write(String(List[UInt8](value.buf, 0)))
        except:
            # TODO: propper logging
            print("could not save zoneinfo to file")
            pass

    fn get(self, key: StringLiteral) -> OptionalReg[Offset]:
        """Get a value from the file.

        Args:
            key: The tz_str.

        Returns:
            An Optional `Offset`.
        """
        var index = hash(key)
        try:
            var value: UInt8
            with open(self._file, "rb") as f:
                _ = f.seek(index * 8)
                value = f.read_bytes(1)
            return Offset(value)
        except:
            return None

    fn __del__(owned self):
        """Delete the file."""
        try:
            import os

            os.remove(self._file)
        except:
            # TODO: propper logging
            print("could not delete zoneinfo file")
            pass


@value
struct ZoneInfoMem32(CollectionElement):
    """`ZoneInfo` that lives in memory. For zones that have DST."""

    var _zones: Dict[StringLiteral, UInt32]

    fn __init__(inout self):
        """Construct a `ZoneInfoMem32`."""

        self._zones = Dict[StringLiteral, UInt32]()

    @always_inline
    fn add(inout self, key: StringLiteral, value: ZoneDST):
        """Add a value to `ZoneInfoMem32`.

        Args:
            key: The tz_str.
            value: Offset.
        """

        self._zones[key] = value.buf

    @always_inline
    fn get(self, key: StringLiteral) -> OptionalReg[ZoneDST]:
        """Get value from `ZoneInfoMem32`.

        Args:
            key: The tz_str.

        Returns:
            An Optional `ZoneDST`.
        """

        var value = self._zones.get(key)
        if not value:
            return None
        return ZoneDST(value.unsafe_take())


@value
struct ZoneInfoMem8(CollectionElement):
    """`ZoneInfo` that lives in memory. For zones that have no DST."""

    var _zones: Dict[StringLiteral, UInt8]

    fn __init__(inout self):
        """Construct a `ZoneInfoMem8`."""
        self._zones = Dict[StringLiteral, UInt8]()

    @always_inline
    fn add(inout self, key: StringLiteral, value: Offset):
        """Add a value to `ZoneInfoMem8`.

        Args:
            key: The tz_str.
            value: Offset.
        """
        self._zones[key] = value.buf

    @always_inline
    fn get(self, key: StringLiteral) -> OptionalReg[Offset]:
        """Get value from `ZoneInfoMem8`.

        Args:
            key: The tz_str.

        Returns:
            An Optional `Offset`.
        """
        var value = self._zones.get(key)
        if not value:
            return None
        return Offset(value.unsafe_take())


# TODO
# fn _parse_iana_zonenow(
#     inout dst_zones: ZoneInfoMem32, inout no_dst_zones: ZoneInfoMem8
# ) raises:
#     pass

# TODO
# fn _parse_iana_dst_transitions(
#     inout dst_zones: ZoneInfoMem32, inout no_dst_zones: ZoneInfoMem8
# ) raises:
#     pass

# TODO
# @always_inline
# fn _parse_iana_leapsecs(
#     text: PythonObject,
# ) raises -> List[(UInt8, UInt8, UInt16)]:
#     var leaps = List[(UInt8, UInt8, UInt16)]()
#     var index = 0
#     while True:
#         var found = text.find("      #", index)
#         if found == -1:
#             break

#         var endday = text.find(" ", found + 2)
#         var day: UInt8 = atol(text.__getitem__(found + 2, endday))

#         var month: UInt8 = 0
#         if text.__getitem__(endday, endday + 3) == "Jan":
#             month = 1
#         elif text.__getitem__(endday, endday + 3) == "Jul":
#             month = 7
#         if month == 0:
#             raise Error("month not found")

#         var year: UInt16 = atol(text.__getitem__(endday + 3, endday + 7))
#         leaps.append((day, month, year))
#     return leaps


@register_passable("trivial")
struct Leapsecs:
    """Leap seconds added to UTC to keep in sync with [IAT](
    https://en.wikipedia.org/wiki/International_Atomic_Time).
    """

    var day: UInt8
    """Day in which the leap second was added."""
    var month: UInt8
    """Month in which the leap second was added."""
    var year: UInt16
    """Year in which the leap second was added."""

    fn __init__(inout self, day: Int, month: Int, year: Int):
        """Construct an `Leapsecs` from values.

        Args:
            day: Day.
            month: Month.
            year: Year.
        """
        self.day = day
        self.month = month
        self.year = year

    fn __init__(inout self, values: Tuple[Int, Int, Int], /):
        """Construct an `Leapsecs` from values.

        Args:
            values: Tuple.
        """

        self = Self(values[0], values[1], values[2])


@always_inline
fn get_leapsecs() -> Optional[List[Leapsecs]]:
    """Get the leap seconds added to UTC.

    Returns:
        A list of tuples (day, month, year) of leapseconds.
    """
    # try:
    #     # TODO: maybe some policy that only if x amount
    #     of years have passed since latest hardcoded value
    #     from python import Python

    #     var requests = Python.import_module("requests")
    #     var secs = requests.get(
    #         "https://raw.githubusercontent.com/eggert/tz/main/leap-seconds.list"
    #     )
    #     var leapsecs = _parse_iana_leapsecs(secs.text)
    #     return leapsecs
    # except:
    #    pass
    from ._lists import leapsecs

    return leapsecs


trait ZoneStorageDST(CollectionElement):
    """Trait that defines ZoneInfo storage structs."""

    fn __init__(inout self):
        """Construct a `ZoneInfo`."""
        ...

    fn add(inout self, key: StringLiteral, value: ZoneDST):
        """Add a value to `ZoneInfo`.

        Args:
            key: The tz_str.
            value: ZoneDST.
        """
        ...

    fn get(self, key: StringLiteral) -> OptionalReg[ZoneDST]:
        """Get value from `ZoneInfo`.

        Args:
            key: The tz_str.

        Returns:
            An Optional `ZoneDST`.
        """
        ...


trait ZoneStorageNoDST(CollectionElement):
    """Trait that defines ZoneInfo storage structs."""

    fn __init__(inout self):
        """Construct a `ZoneInfo`."""
        ...

    fn add(inout self, key: StringLiteral, value: Offset):
        """Add a value to `ZoneInfo`.

        Args:
            key: The tz_str.
            value: Offset.
        """
        ...

    fn get(self, key: StringLiteral) -> OptionalReg[Offset]:
        """Get value from `ZoneInfo`.

        Args:
            key: The tz_str.

        Returns:
            An Optional `Offset`.
        """
        ...


@value
struct ZoneInfo[T: ZoneStorageDST, A: ZoneStorageNoDST]:
    """ZoneInfo.

    Parameters:
        T: The type of storage for timezones with
            Daylight Saving Time.
        A: The type of storage for timezones with
            no Daylight Saving Time.
    """

    var with_dst: T
    """Zoneinfo for Zones with Daylight Saving Time."""
    var with_no_dst: A
    """Zoneinfo for Zones with no Daylight Saving Time."""


fn get_zoneinfo[
    T: ZoneStorageDST = ZoneInfoMem32, A: ZoneStorageNoDST = ZoneInfoMem8
](owned timezones: List[StringLiteral] = List[StringLiteral]()) -> Optional[
    ZoneInfo[T, A]
]:
    """Get all zoneinfo available. First tries to get it
    from the OS, then from the internet, then falls back
    on hardcoded values.

    Parameters:
        T: The type of storage for timezones with
            Daylight Saving Time.
        A: The type of storage for timezones with
            no Daylight Saving Time.

    Args:
        timezones: A list of the timezones to look for info. If the
            list is empty, it defaults to using the hardcoded tz_list
            in `._lists.mojo`.

    Returns:
        Optional ZoneInfo.

    - TODO: this should get zoneinfo from the OS it's compiled in
    - TODO: should have a fallback to hardcoded
    - TODO: this should use IANA's https://raw.githubusercontent.com/eggert/tz/main/zonenow.tab
        - but "# The format of this table is experimental, and may change in future versions."
        - Excerpt:
        ```text
        # -10
        XX	-1732-14934	Pacific/Tahiti	Tahiti; Cook Islands
        #
        # -10/-09 - HST / HDT (North America DST)
        XX	+515248-1763929	America/Adak	western Aleutians in Alaska ("HST/HDT")
        #
        # -09:30
        XX	-0900-13930	Pacific/Marquesas	Marquesas
        ```
        Meanwhile 2 public APIs can be used https://worldtimeapi.org/api
        and https://timeapi.io/swagger/index.html .
    """
    var dst_zones = T()
    var no_dst_zones = A()
    # if len(timezones) == 0:
    #     from ._lists import tz_list

    #     timezones = List[StringLiteral](tz_list)
    # try:
    #     # TODO: this should get zoneinfo from the OS it's compiled in
    #     # for Linux the files are under /usr/share/zoneinfo
    #     # no idea where they're for Windows or MacOS
    #     pass
    # except:
    #     pass
    # try:
    #     from python import Python

    #     var json = Python.import_module("json")
    #     var requests = Python.import_module("requests")
    #     var datetime = Python.import_module("datetime")
    #     # var text = requests.get("https://worldtimeapi.org/api/timezone").text
    #     # var tz_list = json.loads(text)

    #     for item in timezones:
    #         var tz = requests.get("https://timeapi.io/TimeZone/" + item[]).text
    #         var data = json.loads(tz)
    #         var utc_offset = data["standardUtcOffset"]["seconds"] // 60
    #         var h = int(utc_offset // 60)
    #         var m = int(utc_offset % 60)
    #         var sign = 1 if utc_offset >= 0 else -1

    #         var dst_start: PythonObject = ""
    #         var dst_end: PythonObject = ""
    #         if not data["hasDayLightSaving"]:
    #             _ = h, m, sign
    #             # TODO: somehow force cast python object to StringLiteral
    #             no_dst_zones.add(item[], Offset(abs(h), abs(m), sign))
    #             continue
    #         # -1 is to avoid Z timezone designation that
    #         # python's datetime doesn't like
    #         dst_start = data["dstInterval"]["dstStart"].__getitem__(0, -1)
    #         dst_end = data["dstInterval"]["dstEnd"].__getitem__(0, -1)

    #         var dt_start = datetime.datetime(dst_start)
    #         var month_start = UInt8(int(dst_start.month))
    #         var dow_start = UInt8(int(dt_start.weekday()))
    #         var eom_start = UInt8(0 if dt_start <= 15 else 1)
    #         var week_start = 0  # TODO
    #         var h_start = UInt8(int(dt_start.hour))
    #         var dt_end = datetime.datetime(dst_end)
    #         var month_end = UInt8(int(dst_end.month))
    #         var week_end = 0  # TODO
    #         var h_end = UInt8(int(dt_end.hour))
    #         var dow_end = UInt8(int(dt_end.weekday()))
    #         var eom_end = UInt8(0 if dt_end <= 15 else 1)

    #         # TODO: somehow force cast python object to StringLiteral
    #         dst_zones.add(
    #             item[],
    #             ZoneDST(
    #                 TzDT(
    #                     month_start, dow_start, eom_start, week_start, h_start
    #                 ),
    #                 TzDT(month_end, dow_end, eom_end, week_end, h_end),
    #                 Offset(abs(h), abs(m), sign),
    #             ),
    #         )
    #     return ZoneInfo(dst_zones, no_dst_zones)
    # except:
    #     pass
    # TODO: fallback to hardcoded
    return None


fn offset_at(
    owned with_dst: OptionalReg[ZoneDST],
    year: UInt16,
    month: UInt8,
    day: UInt8,
    hour: UInt8,
    minute: UInt8,
    second: UInt8,
) -> OptionalReg[Offset]:
    """Return the UTC offset for the `TimeZone` at the given date
    if it has DST.

    Args:
        with_dst: Optional zone with DST.
        year: Year.
        month: Month.
        day: Day.
        hour: Hour.
        minute: Minute.
        second: Second.

    Returns:
        - offset_h: Offset for the hour: [0, 15].
        - offset_m: Offset for the minute: {0, 30, 45}.
        - sign: Sign of the offset: {1, -1}.
    """
    if not with_dst:
        return None
    var zone = with_dst.value()
    var items = zone.from_hash()
    var dst_start = items[0]
    var dst_end = items[1]
    var offset = items[2]
    var sign: UInt8 = offset.sign
    var m: UInt8 = offset.minute
    var dst_h = offset.hour + 1
    var std_m = m
    var dst_m = m
    # if it's a weird tz
    if offset.minute == 3:
        if (offset.buf & 0b1) == 0:  # "Australia/Lord_Howe"
            dst_h = 0
            std_m = 0
            dst_m = 30
        elif (offset.buf & 0b1) == 1:  # "Antarctica/Troll"
            dst_h += 1
            std_m = 0
            dst_m = 0

    var std = Offset(offset.hour, std_m, sign)
    var dst = Offset(dst_h, dst_m, sign)

    fn eval_dst(dst_st: Bool, data: TzDT) -> Offset:
        var is_end_mon = data.eomon == 1
        var maxdays = _cal.max_days_in_month(year, month)
        var iterable = range(0, maxdays, step=1)
        if is_end_mon:
            iterable = range(maxdays - 1, -1, step=-1)

        var dow_target = data.dow
        var dow = _cal.dayofweek(year, month, day)
        var amnt_weeks_target = data.week
        var is_later = hour > data.hour and minute > 0 and second >= 0
        var accum: UInt8 = 0
        for i in iterable:
            if _cal.dayofweek(year, month, i) == dow_target:
                if accum != amnt_weeks_target:
                    accum += 1
                    continue
            var is_less = dow < dow_target
            var is_more = dow > dow_target
            var is_start_mon = not is_end_mon
            var is_dst_start_and_dow_is_less = dst_st and is_less
            var is_dst_end_and_dow_is_more = not dst_st and is_more

            if is_start_mon and is_dst_start_and_dow_is_less:
                return std
            elif is_start_mon and is_dst_end_and_dow_is_more:
                return std
            elif is_end_mon and is_dst_start_and_dow_is_less:
                return std
            elif is_end_mon and is_dst_end_and_dow_is_more:
                return std
            elif is_start_mon and not is_later:
                return std
            elif is_end_mon and is_later:
                return std
            break
        return dst

    if month == dst_start.month:
        return eval_dst(True, dst_start)
    elif month == dst_end.month:
        return eval_dst(False, dst_end)
    elif month > dst_start.month and month < dst_end.month:
        return dst
    return std
