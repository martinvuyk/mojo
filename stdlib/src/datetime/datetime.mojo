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
"""Nanosecond resolution `DateTime` module.

Notes:
    - IANA is supported: [`TimeZone` and DST data sources](
        http://www.iana.org/time-zones/repository/tz-link.html).
        [List of TZ identifiers (`tz_str`)](
        https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
"""
from time import time
from utils import Variant
from collections.optional import Optional

from .timezone import (
    TimeZone,
    ZoneInfo,
    ZoneInfoMem32,
    ZoneInfoMem8,
    ZoneStorageDST,
    ZoneStorageNoDST,
)
from .calendar import Calendar, UTCCalendar, PythonCalendar, CalendarHashes
import .dt_str

alias _calendar = PythonCalendar
alias _cal_hash = CalendarHashes(64)
alias _max_delta = UInt16(~UInt64(0) // (365 * 24 * 60 * 60 * 1_000_000_000))
"""Maximum year delta that fits in a UInt64 for a 
Gregorian calendar with year = 365 d * 24 h, 60 min, 60 s, 10^9 ns"""


trait _IntCollect(Intable, CollectionElement):
    ...


@value
# @register_passable("trivial")
struct DateTime[
    dst_storage: ZoneStorageDST = ZoneInfoMem32,
    no_dst_storage: ZoneStorageNoDST = ZoneInfoMem8,
    iana: Bool = True,
    pyzoneinfo: Bool = True,
    native: Bool = False,
](Hashable, Stringable):
    """Custom `Calendar` and `TimeZone` may be passed in.
    By default, it uses `PythonCalendar` which is a Gregorian
    calendar with its given epoch and max year:
    [0001-01-01, 9999-12-31]. Default `TimeZone` is UTC.

    Parameters:
        dst_storage: The type of storage to use for ZoneInfo
            for zones with Dailight Saving Time. Default Memory.
        no_dst_storage: The type of storage to use for ZoneInfo
            for zones with no Dailight Saving Time. Default Memory.
        iana: Whether timezones from the [IANA database](
            http://www.iana.org/time-zones/repository/tz-link.html)
            are used. It defaults to using all available timezones,
            if getting them fails at compile time, it tries using
            python's zoneinfo if pyzoneinfo is set to True, otherwise
            it uses the offsets as is, no daylight saving or
            special exceptions. [List of TZ identifiers](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
        pyzoneinfo: Whether to use python's zoneinfo and
            datetime to get full IANA support.
        native: (fast, partial IANA support) Whether to use a native Dict
            with the current timezones from the [List of TZ identifiers](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
            at the time of compilation (for now they're hardcoded
            at stdlib release time, in the future it should get them
            from the OS). If it fails at compile time, it defaults to
            using the given offsets when the timezone was constructed.

    - Max Resolution:
        - year: Up to year 65_536.
        - month: Up to month 256.
        - day: Up to day 256.
        - hour: Up to hour 256.
        - minute: Up to minute 256.
        - second: Up to second 256.
        - m_second: Up to m_second 65_536.
        - u_second: Up to u_second 65_536.
        - n_second: Up to n_second 65_536.
        - hash: 64 bits.

    - Notes:
        - The default hash that is used for logical and bitwise
            operations has only microsecond resolution.
        - The Default `DateTime` hash has only Microsecond resolution.
    """

    var year: UInt16
    """Year."""
    var month: UInt8
    """Month."""
    var day: UInt8
    """Day."""
    var hour: UInt8
    """Hour."""
    var minute: UInt8
    """Minute."""
    var second: UInt8
    """Second."""
    var m_second: UInt16
    """M_second."""
    var u_second: UInt16
    """U_second."""
    var n_second: UInt16
    """N_second."""
    # TODO: tz and calendar should be references
    alias _tz = TimeZone[dst_storage, no_dst_storage, iana, pyzoneinfo, native]
    var tz: Self._tz
    """Tz."""
    var calendar: Calendar
    """Calendar."""

    fn __init__[
        T1: _IntCollect = Int,
        T2: _IntCollect = Int,
        T3: _IntCollect = Int,
        T4: _IntCollect = Int,
        T5: _IntCollect = Int,
        T6: _IntCollect = Int,
        T7: _IntCollect = Int,
        T8: _IntCollect = Int,
        T9: _IntCollect = Int,
    ](
        inout self,
        owned year: Optional[T1] = None,
        owned month: Optional[T2] = None,
        owned day: Optional[T3] = None,
        owned hour: Optional[T4] = None,
        owned minute: Optional[T5] = None,
        owned second: Optional[T6] = None,
        owned m_second: Optional[T7] = None,
        owned u_second: Optional[T8] = None,
        owned n_second: Optional[T9] = None,
        owned tz: Self._tz = Self._tz(),
        owned calendar: Calendar = _calendar,
    ):
        """Construct a `DateTime` from valid values.

        Parameters:
            T1: Any type that is Intable and CollectionElement.
            T2: Any type that is Intable and CollectionElement.
            T3: Any type that is Intable and CollectionElement.
            T4: Any type that is Intable and CollectionElement.
            T5: Any type that is Intable and CollectionElement.
            T6: Any type that is Intable and CollectionElement.
            T7: Any type that is Intable and CollectionElement.
            T8: Any type that is Intable and CollectionElement.
            T9: Any type that is Intable and CollectionElement.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.
            tz: Tz.
            calendar: Calendar.
        """
        self.year = int(year.take()) if year else int(calendar.min_year)
        self.month = int(month.take()) if month else int(calendar.min_month)
        self.day = int(day.take()) if day else int(calendar.min_day)
        self.hour = int(hour.take()) if hour else int(calendar.min_hour)
        self.minute = int(minute.take()) if minute else int(calendar.min_minute)
        self.second = int(second.take()) if second else int(calendar.min_second)
        self.m_second = int(m_second.take()) if m_second else int(
            calendar.min_milisecond
        )
        self.u_second = int(u_second.take()) if u_second else int(
            calendar.min_microsecond
        )
        self.n_second = int(n_second.take()) if n_second else int(
            calendar.min_nanosecond
        )
        self.tz = tz
        self.calendar = calendar

    @staticmethod
    fn _from_overflow(
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        hours: Int = 0,
        minutes: Int = 0,
        seconds: Int = 0,
        m_seconds: Int = 0,
        u_seconds: Int = 0,
        n_seconds: Int = 0,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from possibly overflowing values."""
        var ns = Self._from_n_seconds(n_seconds, tz, calendar)
        var us = Self._from_u_seconds(u_seconds, tz, calendar)
        var ms = Self._from_m_seconds(m_seconds, tz, calendar)
        var s = Self.from_seconds(seconds, tz, calendar)
        var m = Self._from_minutes(minutes, tz, calendar)
        var h = Self._from_hours(hours, tz, calendar)
        var d = Self._from_days(days, tz, calendar)
        var mon = Self._from_months(months, tz, calendar)
        var y = Self._from_years(years, tz, calendar)

        y.year = 0 if years == 0 else y.year

        for dt in List(mon, d, h, m, s, ms, us, ns):
            if dt[].year != calendar.min_year:
                y.year += dt[].year
        y.month = mon.month
        for dt in List(d, h, m, s, ms, us, ns):
            if dt[].month != calendar.min_month:
                y.month += dt[].month
        y.day = d.day
        for dt in List(h, m, s, ms, us, ns):
            if dt[].day != calendar.min_day:
                y.day += dt[].day
        y.hour = h.hour
        for dt in List(m, s, ms, us, ns):
            if dt[].hour != calendar.min_hour:
                y.hour += dt[].hour
        y.minute = m.minute
        for dt in List(s, ms, us, ns):
            if dt[].minute != calendar.min_minute:
                y.minute += dt[].minute
        y.second = s.second
        for dt in List(ms, us, ns):
            if dt[].second != calendar.min_second:
                y.second += dt[].second
        y.m_second = ms.m_second
        for dt in List(us, ns):
            if dt[].m_second != calendar.min_milisecond:
                y.m_second += dt[].m_second
        y.u_second = us.u_second
        if ns.u_second != calendar.min_microsecond:
            y.u_second += ns.u_second
        y.n_second = ns.n_second
        return y

    fn replace(
        owned self,
        *,
        owned year: Optional[UInt16] = None,
        owned month: Optional[UInt8] = None,
        owned day: Optional[UInt8] = None,
        owned hour: Optional[UInt8] = None,
        owned minute: Optional[UInt8] = None,
        owned second: Optional[UInt8] = None,
        owned m_second: Optional[UInt16] = None,
        owned u_second: Optional[UInt16] = None,
        owned n_second: Optional[UInt16] = None,
        owned tz: Optional[Self._tz] = None,
        owned calendar: Optional[Calendar] = None,
    ) -> Self:
        """Replace with give value/s.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: Milisecond.
            u_second: Microsecond.
            n_second: Nanosecond.
            tz: Tz.
            calendar: Calendar to change to, distance from epoch
                is calculated and the new Self has that same
                distance from the new Calendar's epoch.

        Returns:
            Self.
        """

        if year:
            self.year = year.take()
        if month:
            self.month = month.take()
        if day:
            self.day = day.take()
        if hour:
            self.hour = hour.take()
        if minute:
            self.minute = minute.take()
        if second:
            self.second = second.take()
        if m_second:
            self.m_second = m_second.take()
        if u_second:
            self.u_second = u_second.take()
        if n_second:
            self.n_second = n_second.take()
        if tz:
            self.tz = tz.take()
        if calendar:
            self.calendar = calendar.take()
        return self

    fn to_calendar(owned self, calendar: Calendar) -> Self:
        """Translates the `DateTime`'s values to be on the same
        offset since it's current calendar's epoch to the new
        calendar's epoch.

        Args:
            calendar: The new calendar.

        Returns:
            Self.
        """
        var year = self.year
        var tmpcal = self.calendar.from_year(year)
        self.calendar = tmpcal
        var ns = self.n_seconds_since_epoch()
        self.year = calendar.min_year
        self.month = calendar.min_month
        self.day = calendar.min_day
        self.hour = calendar.min_hour
        self.minute = calendar.min_minute
        self.second = calendar.min_second
        self.calendar = calendar
        return self.add(years=int(year), n_seconds=int(ns))

    fn to_utc(owned self) -> Self:
        """Returns a new instance of `Self` transformed to UTC. If
        `self.tz` is UTC it returns early.

        Returns:
            Self.
        """
        alias TZ_UTC = Self._tz()
        if self.tz == TZ_UTC:
            return self
        var new_self = self
        var offset = self.tz.offset_at(
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
        var of_h = int(offset.hour)
        var of_m = int(offset.minute)
        if offset.sign == -1:
            new_self = self.add(hours=of_h, minutes=of_m)
        else:
            new_self = self.subtract(hours=of_h, minutes=of_m)
        new_self.tz = TZ_UTC
        return new_self

    fn from_utc(owned self, tz: Self._tz) -> Self:
        """Translate `TimeZone` from UTC. If `self.tz` is UTC
        it returns early.

        Args:
            tz: Timezone to cast to.

        Returns:
            Self.
        """
        alias TZ_UTC = Self._tz()
        if tz == TZ_UTC:
            return self
        var offset = tz.offset_at(
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
        var h = int(offset.hour)
        var m = int(offset.minute)
        var new_self: Self
        if offset.sign == 1:
            new_self = self.add(hours=h, minutes=m)
        else:
            new_self = self.subtract(hours=h, minutes=m)
        var leapsecs = int(
            new_self.calendar.leapsecs_since_epoch(
                new_self.year, new_self.month, new_self.day
            )
        )
        return new_self.add(seconds=leapsecs).replace(tz=tz)

    fn n_seconds_since_epoch(self) -> UInt64:
        """Nanoseconds since the begining of the calendar's epoch.
        Can only represent up to ~ 580 years since epoch start.

        Returns:
            The amount.
        """
        return self.calendar.n_seconds_since_epoch(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.m_second,
            self.u_second,
            self.n_second,
        )

    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            The amount.
        """
        return self.calendar.seconds_since_epoch(
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )

    fn delta_ns(self, other: Self) -> (UInt64, UInt64, UInt16, UInt8):
        """Calculates the nanoseconds for `self` and other, creating
        a reference calendar to keep nanosecond resolution.

        Args:
            other: Other.

        Returns:
            - self_ns: Nanoseconds from `self` to created temp calendar.
            - other_ns: Nanoseconds from other to created temp calendar.
            - overflow: the amount of years added / subtracted from `self`
                to make the temp calendar. This occurs if the difference
                in years is bigger than ~ 580 (Gregorian years).
            - sign: {1, -1} if the overflow was added or subtracted.
        """
        var s = self
        var o = other
        if s.tz != o.tz:
            s = s.to_utc()
            o = o.to_utc()

        var overflow: UInt16 = 0
        var sign: UInt8 = 1
        var year = s.year
        if s.year < o.year:
            sign = -1
            while o.year - year > _max_delta:
                year -= _max_delta
                overflow += _max_delta
        else:
            while year - o.year > _max_delta:
                year -= _max_delta
                overflow += _max_delta

        var cal = Calendar.from_year(year)
        var self_ns = s.replace(calendar=cal).n_seconds_since_epoch()
        var other_ns = o.replace(calendar=cal).n_seconds_since_epoch()
        return self_ns, other_ns, overflow, sign

    fn add(
        owned self,
        *,
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        hours: Int = 0,
        minutes: Int = 0,
        seconds: Int = 0,
        m_seconds: Int = 0,
        u_seconds: Int = 0,
        n_seconds: Int = 0,
    ) -> Self:
        """Recursively evaluated function to build a valid `DateTime`
        according to its calendar.

        Args:
            years: Years.
            months: Months.
            days: Days.
            hours: Hours.
            minutes: Minutes.
            seconds: Seconds.
            m_seconds: Miliseconds.
            u_seconds: Microseconds.
            n_seconds: Nanoseconds.

        Returns:
            Self.

        Notes:
            On overflow, the `DateTime` starts from the beginning of the
            calendar's epoch and keeps evaluating until valid.
        """
        var dt = self._from_overflow(
            int(self.year) + years,
            int(self.month) + months,
            int(self.day) + days,
            int(self.hour) + hours,
            int(self.minute) + minutes,
            int(self.second) + seconds,
            int(self.m_second) + m_seconds,
            int(self.u_second) + u_seconds,
            int(self.n_second) + n_seconds,
            self.tz,
            self.calendar,
        )
        var minyear = dt.calendar.min_year
        var maxyear = dt.calendar.max_year
        if dt.year > maxyear:
            dt = dt.replace(year=minyear).add(years=int(dt.year - maxyear))
        var minmon = dt.calendar.min_month
        var maxmon = dt.calendar.max_month
        if dt.month > maxmon:
            dt = dt.replace(month=minmon).add(
                years=1, months=int(dt.month - maxmon)
            )
        var minday = dt.calendar.min_day
        var maxday = dt.calendar.max_days_in_month(dt.year, dt.month)
        if dt.day > maxday:
            dt = dt.replace(day=minday).add(months=1, days=int(dt.day - maxday))
        var minhour = dt.calendar.min_hour
        var maxhour = dt.calendar.max_hour
        if dt.hour > maxhour:
            dt = dt.replace(hour=minhour).add(
                days=1, hours=int(dt.hour - maxhour)
            )
        var minmin = dt.calendar.min_minute
        var maxmin = dt.calendar.max_minute
        if dt.minute > maxmin:
            dt = dt.replace(minute=minmin).add(
                hours=1, minutes=int(dt.minute - maxmin)
            )
        var minsec = dt.calendar.min_second
        var maxsec = dt.calendar.max_second(
            dt.year, dt.month, dt.day, dt.hour, dt.minute
        )
        if dt.second > maxsec:
            dt = dt.replace(second=minsec).add(
                minutes=1, seconds=int(dt.second - maxsec)
            )
        var minmsec = dt.calendar.min_milisecond
        var maxmsec = dt.calendar.max_milisecond
        if dt.m_second > maxmsec:
            dt = dt.replace(m_second=minmsec).add(
                seconds=1, m_seconds=int(dt.m_second - maxmsec)
            )
        var minusec = dt.calendar.min_microsecond
        var maxusec = dt.calendar.max_microsecond
        if dt.u_second > maxusec:
            dt = dt.replace(u_second=minusec).add(
                m_seconds=1, u_seconds=int(dt.u_second - maxusec)
            )
        var minnsec = dt.calendar.min_nanosecond
        var maxnsec = dt.calendar.max_nanosecond
        if dt.n_second > maxnsec:
            dt = dt.replace(n_second=minnsec).add(
                u_seconds=1, n_seconds=int(dt.n_second - maxnsec)
            )
        return dt

    fn subtract(
        owned self,
        *,
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        hours: Int = 0,
        minutes: Int = 0,
        seconds: Int = 0,
        m_seconds: Int = 0,
        u_seconds: Int = 0,
        n_seconds: Int = 0,
    ) -> Self:
        """Recursively evaluated function to build a valid `DateTime`
        according to its calendar.

        Args:
            years: Years.
            months: Months.
            days: Days.
            hours: Hours.
            minutes: Minutes.
            seconds: Seconds.
            m_seconds: Miliseconds.
            u_seconds: Microseconds.
            n_seconds: Nanoseconds.

        Returns:
            Self.

        Notes:
            On overflow, the `DateTime` goes to the end of the
            calendar's epoch and keeps evaluating until valid.
        """
        var dt = self._from_overflow(
            int(self.year) - years,
            int(self.month) - months,
            int(self.day) - days,
            int(self.hour) - hours,
            int(self.minute) - minutes,
            int(self.second) - seconds,
            int(self.m_second) - m_seconds,
            int(self.u_second) - u_seconds,
            int(self.n_second) - n_seconds,
            self.tz,
            self.calendar,
        )
        var minyear = dt.calendar.min_year
        var maxyear = dt.calendar.max_year
        if dt.year < minyear:
            dt = dt.replace(year=maxyear).subtract(years=int(minyear - dt.year))
        var minmonth = dt.calendar.min_month
        var maxmonth = dt.calendar.max_month
        if dt.month < minmonth:
            dt = dt.replace(month=maxmonth).subtract(
                years=1, months=int(minmonth - dt.month)
            )
        var minday = dt.calendar.min_day
        if dt.day < minday:
            dt = dt.subtract(months=1)
            var prev_day = dt.calendar.max_days_in_month(dt.year, dt.month - 1)
            dt = dt.replace(day=prev_day).subtract(days=int(minday - dt.day))
        var minhour = dt.calendar.min_hour
        var maxhour = dt.calendar.max_hour
        if dt.hour < minhour:
            dt = dt.replace(hour=maxhour).subtract(
                days=1, hours=int(minhour - dt.hour)
            )
        var minmin = dt.calendar.min_minute
        var maxmin = dt.calendar.max_minute
        if dt.minute < minmin:
            dt = dt.replace(minute=maxmin).subtract(
                hours=1, minutes=int(minmin - dt.minute)
            )
        var minsec = dt.calendar.min_second
        if dt.second < minsec:
            var sec = dt.calendar.max_second(
                dt.year, dt.month, dt.day, dt.hour, dt.minute
            )
            dt = dt.replace(second=sec).subtract(
                minutes=1, seconds=int(minsec - dt.second)
            )
        var minmsec = dt.calendar.min_milisecond
        var maxmsec = dt.calendar.max_milisecond
        if dt.m_second < minmsec:
            dt = dt.replace(m_second=maxmsec).subtract(
                seconds=1, m_seconds=int(minmsec - dt.m_second)
            )
        var minusec = dt.calendar.min_microsecond
        var maxusec = dt.calendar.max_microsecond
        if dt.u_second < minusec:
            dt = dt.replace(u_second=maxusec).subtract(
                m_seconds=1, u_seconds=int(minusec - dt.u_second)
            )
        var minnsec = dt.calendar.min_nanosecond
        var maxnsec = dt.calendar.max_nanosecond
        if dt.n_second < minnsec:
            dt = dt.replace(n_second=maxnsec).subtract(
                u_seconds=1, n_seconds=int(minnsec - dt.n_second)
            )
        return dt

    # @always_inline("nodebug")
    fn add(owned self, other: Self) -> Self:
        """Adds another `DateTime`.

        Args:
            other: Other.

        Returns:
            A `DateTime` with the `TimeZone` and `Calendar` of `self`.
        """
        return self.add(
            years=int(other.year),
            months=int(other.month),
            days=int(other.day),
            hours=int(other.hour),
            minutes=int(other.minute),
            seconds=int(other.second),
            m_seconds=int(other.m_second),
            u_seconds=int(other.u_second),
            n_seconds=int(other.n_second),
        )

    # @always_inline("nodebug")
    fn subtract(owned self, other: Self) -> Self:
        """Subtracts another `DateTime`.

        Args:
            other: Other.

        Returns:
            A `DateTime` with the `TimeZone` and `Calendar` of `self`.
        """
        return self.subtract(
            years=int(other.year),
            months=int(other.month),
            days=int(other.day),
            hours=int(other.hour),
            minutes=int(other.minute),
            seconds=int(other.second),
            m_seconds=int(other.m_second),
            u_seconds=int(other.u_second),
            n_seconds=int(other.n_second),
        )

    # @always_inline("nodebug")
    fn __add__(owned self, other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.add(other)

    # @always_inline("nodebug")
    fn __sub__(owned self, other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.subtract(other)

    # @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self = self.add(other)

    # @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self = self.subtract(other)

    # @always_inline("nodebug")
    fn dayofweek(self) -> UInt8:
        """Calculates the day of the week for a `DateTime`.

        Returns:
            - day: Day of the week: [0, 6] (monday - sunday) (default).
        """
        return self.calendar.dayofweek(self.year, self.month, self.day)

    fn dayofyear(self) -> UInt16:
        """Calculates the day of the year for a `DateTime`.

        Returns:
            - day: Day of the year: [1, 366] (for gregorian calendar).
        """
        return self.calendar.dayofyear(self.year, self.month, self.day)

    fn leapsecs_since_epoch(self) -> UInt32:
        """Cumulative leap seconds since the calendar's epoch start.

        Returns:
            The amount.
        """
        var dt = self.to_utc()
        return dt.calendar.leapsecs_since_epoch(dt.year, dt.month, dt.day)

    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return self.calendar.hash[_cal_hash](
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.m_second,
            self.u_second,
            self.n_second,
        )

    # @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        var s = self
        var o = other
        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        return hash(s) == hash(o)

    # @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        var s = self
        var o = other
        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        return hash(s) != hash(o)

    # @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        var s = self
        var o = other
        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        return hash(s) > hash(o)

    # @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        var s = self
        var o = other
        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        return hash(s) >= hash(o)

    # @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        var s = self
        var o = other
        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        return hash(s) <= hash(o)

    # @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        var s = self
        var o = other
        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        return hash(s) < hash(o)

    # @always_inline("nodebug")
    fn __invert__(self) -> UInt32:
        """Invert.

        Returns:
            Self.
        """
        return ~hash(self)

    # @always_inline("nodebug")
    fn __and__[T: Hashable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return hash(self) & hash(other)

    # @always_inline("nodebug")
    fn __or__[T: Hashable](self, other: T) -> UInt64:
        """Or.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return hash(self) | hash(other)

    # @always_inline("nodebug")
    fn __xor__[T: Hashable](self, other: T) -> UInt64:
        """Xor.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return hash(self) ^ hash(other)

    # @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return hash(self)

    # @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @staticmethod
    fn _from_years(
        years: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from years."""
        var delta = int(calendar.max_year) - years
        if delta > 0:
            if years > int(calendar.min_year):
                return Self(year=years, tz=tz, calendar=calendar)
            return Self._from_years(delta)
        return Self._from_years(int(calendar.max_year) - delta)

    @staticmethod
    # @always_inline("nodebug")
    fn _from_months(
        months: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from months."""
        if months <= int(calendar.max_month):
            return Self(month=months, tz=tz, calendar=calendar)
        var y = months // int(calendar.max_month)
        var rest = months % int(calendar.max_month)
        var dt = Self._from_years(y, tz, calendar)
        dt.month = rest
        return dt

    @staticmethod
    fn _from_days[
        add_leap: Bool = False
    ](
        days: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from days."""
        var minyear = int(calendar.min_year)
        var dt = Self(minyear, tz=tz, calendar=calendar)
        var maxtdays = int(calendar.max_typical_days_in_year)
        var maxposdays = int(calendar.max_possible_days_in_year)
        var years = days // maxtdays
        if years > minyear:
            dt = Self._from_years(years, tz, calendar)
        var maxydays = maxposdays if calendar.is_leapyear(dt.year) else maxtdays
        var day = days
        if add_leap:
            var leapdays = calendar.leapdays_since_epoch(
                dt.year, dt.month, dt.day
            )
            day += int(leapdays)
        if day > maxydays:
            var y = day // maxydays
            day = day % maxydays
            var dt2 = Self._from_years(y, tz, calendar)
            dt.year += dt2.year
        var maxmondays = int(calendar.max_days_in_month(dt.year, dt.month))
        while day > maxmondays:
            day -= maxmondays
            dt.month += 1
            maxmondays = int(calendar.max_days_in_month(dt.year, dt.month))
        dt.day = day
        return dt

    @staticmethod
    fn _from_hours[
        add_leap: Bool = False
    ](
        hours: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from hours."""
        var h = int(calendar.max_hour)
        if hours <= h:
            return Self(hour=hours, tz=tz, calendar=calendar)
        var d = hours // (h + 1)
        var rest = hours % (h + 1)
        var dt = Self._from_days[add_leap](d, tz, calendar)
        dt.hour = rest
        return dt

    @staticmethod
    fn _from_minutes[
        add_leap: Bool = False
    ](
        minutes: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from minutes."""
        var m = int(calendar.max_minute)
        if minutes < m:
            return Self(minute=minutes, tz=tz, calendar=calendar)
        var h = minutes // (m + 1)
        var rest = minutes % (m + 1)
        var dt = Self._from_hours[add_leap](h, tz, calendar)
        dt.minute = rest
        return dt

    @staticmethod
    fn from_seconds[
        add_leap: Bool = False
    ](
        seconds: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from seconds.

        Parameters:
            add_leap: Whether to add the leap seconds and leap days
                since the start of the calendar's epoch.

        Args:
            seconds: Seconds.
            tz: Tz.
            calendar: Calendar.

        Returns:
            Self.
        """
        var s = int(calendar.max_typical_second)
        var minutes = seconds // (s + 1)
        var dt = Self._from_minutes(minutes, tz, calendar)
        var numerator = seconds
        if add_leap:
            var leapsecs = calendar.leapsecs_since_epoch(
                dt.year, dt.month, dt.day
            )
            numerator += int(leapsecs)
        var m = numerator // (s + 1)
        var rest = numerator % (s + 1)
        dt = Self._from_minutes(m, tz, calendar)
        var max_second = int(
            calendar.max_second(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        )
        while rest > max_second:
            rest -= max_second
            dt.minute += 1
            max_second = int(
                calendar.max_second(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute
                )
            )
        dt.second = rest
        return dt

    @staticmethod
    fn _from_m_seconds(
        m_seconds: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from miliseconds."""
        var ms = int(calendar.max_milisecond)
        if m_seconds <= ms:
            return Self(m_second=m_seconds, tz=tz, calendar=calendar)
        var s = m_seconds // (ms + 1)
        var rest = m_seconds % (ms + 1)
        var dt = Self.from_seconds(s, tz, calendar)
        dt.m_second = rest
        return dt

    @staticmethod
    fn _from_u_seconds(
        u_seconds: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from microseconds."""
        var us = int(calendar.max_microsecond)
        if u_seconds <= us:
            return Self(u_second=u_seconds, tz=tz, calendar=calendar)
        var ms = u_seconds // (us + 1)
        var rest = u_seconds % (us + 1)
        var dt = Self._from_m_seconds(ms, tz, calendar)
        dt.u_second = rest
        return dt

    @staticmethod
    # @always_inline("nodebug")
    fn _from_n_seconds(
        n_seconds: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from nanoseconds."""
        var ns = int(calendar.max_nanosecond)
        if n_seconds <= ns:
            return Self(n_second=n_seconds, tz=tz, calendar=calendar)
        var us = n_seconds // (ns + 1)
        var rest = n_seconds % (ns + 1)
        var dt = Self._from_u_seconds(us, tz, calendar)
        dt.n_second = rest
        return dt

    @staticmethod
    # @always_inline("nodebug")
    fn from_unix_epoch[
        add_leap: Bool = False
    ](seconds: Int, tz: Self._tz = Self._tz(),) -> Self:
        """Construct a `DateTime` from the seconds since the Unix Epoch
        1970-01-01. Adding the cumulative leap seconds since 1972
        to the given date.

        Parameters:
            add_leap: Whether to add the leap seconds and leap days
                since the start of the calendar's epoch.

        Args:
            seconds: Seconds.
            tz: Tz.

        Returns:
            Self.
        """
        return Self.from_seconds[add_leap](seconds, tz=tz, calendar=UTCCalendar)

    @staticmethod
    # @always_inline("nodebug")
    fn now(
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a datetime from `time.now()`.

        Args:
            tz: `TimeZone` to replace UTC.
            calendar: Calendar to replace the UTCCalendar with.

        Returns:
            Self.
        """
        var ns = time.now()
        var us: UInt16 = ns // 1_000
        var ms: UInt16 = ns // 1_000_000
        var s = ns // 1_000_000_000
        var dt = Self.from_unix_epoch(s, tz).replace(calendar=calendar)
        return dt.replace(m_second=ms, u_second=us, n_second=UInt16(ns))

    # @always_inline("nodebug")
    fn strftime[format_str: StringLiteral](self) -> String:
        """Formats time into a `String`.

        Parameters:
            format_str: Format string.

        Returns:
            The formatted string.

        - TODO
            - localization.
        """
        return dt_str.strftime[format_str](
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )

    # @always_inline("nodebug")
    fn strftime(self, fmt: String) -> String:
        """Formats time into a `String`.

        Args:
            fmt: Format string.

        Returns:
            The formatted string.

        - TODO
            - localization.
        """
        return dt_str.strftime(
            fmt,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )

    # @always_inline("nodebug")
    fn __format__(self, fmt: String) -> String:
        """Format.

        Args:
            fmt: Format string.

        Returns:
            String.
        """
        return self.strftime(fmt)

    # @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant formatted `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_TZD`
        -> `1970-01-01T00:00:00+00:00` .

        Parameters:
            iso: The IsoFormat.

        Returns:
            String.
        """
        var date = (int(self.year), int(self.month), int(self.day))
        var hour = (int(self.hour), int(self.minute), int(self.second))
        var time = dt_str.to_iso[iso](
            date[0], date[1], date[2], hour[0], hour[1], hour[2]
        )
        return time + self.tz.to_iso()

    @staticmethod
    # @always_inline("nodebug")
    fn strptime[
        format_str: StringLiteral,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Parse a `DateTime` from a  `String`.

        Parameters:
            format_str: The format string.
            tz: The `TimeZone` to cast the result to.
            calendar: The Calendar to cast the result to.

        Args:
            s: The string.

        Returns:
            An Optional Self.
        """
        var parsed = dt_str.strptime[format_str](s)
        if not parsed:
            return None
        var p = parsed.take()
        return Self(
            p.year,
            p.month,
            p.day,
            p.hour,
            p.minute,
            p.second,
            p.m_second,
            p.u_second,
            p.n_second,
            tz,
            calendar,
        )

    @staticmethod
    # @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        tz: Optional[Self._tz] = None,
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a datetime from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The IsoFormat to parse.
            tz: Optional timezone to transform the result into
                (taking into account that the format may return with a `TimeZone`).
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional Self.
        """
        try:
            var p = dt_str.from_iso[
                iso, dst_storage, no_dst_storage, iana, pyzoneinfo, native
            ](s)
            var dt = Self(
                p[0], p[1], p[2], p[3], p[4], p[5], tz=p[6], calendar=calendar
            )
            if tz:
                var t = tz.value()
                if t != dt.tz:
                    return dt.to_utc().from_utc(t)
            return dt
        except:
            return None

    @staticmethod
    # @always_inline("nodebug")
    fn from_hash(
        value: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from a hash made by it.
        Nanoseconds are set to the calendar's minimum.

        Args:
            value: The value to parse.
            tz: The `TimeZone` to designate to the result.
            calendar: The Calendar to designate to the result.

        Returns:
            Self.
        """
        var d = calendar.from_hash(value)
        var ns = calendar.min_nanosecond
        return Self(
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], ns, tz, calendar
        )
