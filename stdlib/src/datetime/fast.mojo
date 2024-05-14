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
"""Fast implementations of `DateTime` module. All assume no leap seconds or
years.

- `DateTime64`:
    - This is a "normal" `DateTime` with milisecond resolution.
- `DateTime32`:
    - This is a "normal" `DateTime` with minute resolution.
- `DateTime16`:
    - This is a `DateTime` with hour resolution, it can be used as a 
    year, dayofyear, hour representation.
- `DateTime8`:
    - This is a `DateTime` with hour resolution, it can be used as a 
    dayofweek, hour representation.
- Notes:
    - The caveats of each implementation are better explained in 
    each struct's docstrings.
"""
from time import time

from .calendar import Calendar, UTCFastCal, CalendarHashes
import .dt_str

alias _calendar = UTCFastCal
alias _cal_h64 = CalendarHashes(64)
alias _cal_h32 = CalendarHashes(32)
alias _cal_h16 = CalendarHashes(16)
alias _cal_h8 = CalendarHashes(8)


trait _IntCollect(Intable, CollectionElement):
    ...


@register_passable("trivial")
struct DateTime64(Hashable, Stringable):
    """Fast `DateTime64` struct. This is a "normal"
    `DateTime` with milisecond resolution. Uses
    given calendar's epoch and other params at
    build time. Assumes all instances have the same
    timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        - year: Up to year 134_217_728.
        - month: Up to month 32.
        - day: Up to day 32.
        - hour: Up to hour 32.
        - minute: Up to minute 64.
        - second: Up to second 64.
        - m_second: Up to m_second 1024.

    - Milisecond Resolution (Gregorian as reference):
        - year: Up to year 584_942_417 since calendar epoch.

    - Notes:
        - Once methods that alter the underlying m_seconds
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var m_second: UInt64
    """Milisecond."""
    var hash: UInt64
    """Hash."""

    fn __init__[
        T: _IntCollect = UInt16, A: _IntCollect = UInt8, B: _IntCollect = UInt64
    ](
        inout self,
        year: Optional[T] = None,
        month: Optional[A] = None,
        day: Optional[A] = None,
        hour: Optional[A] = None,
        minute: Optional[A] = None,
        second: Optional[A] = None,
        m_second: Optional[T] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[B] = None,
    ):
        """Construct a `DateTime64` from valid values.
        UTCCalendar is the default.

        Parameters:
            T: Any Intable Collectable type.
            A: Any Intable Collectable type.
            B: Any Intable Collectable type.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = int(year.value()[]) if year else int(calendar.min_year)
        var mon = int(month.value()[]) if month else int(calendar.min_month)
        var d = int(day.value()[]) if day else int(calendar.min_day)
        var h = int(hour.value()[]) if hour else int(calendar.min_hour)
        var m = int(minute.value()[]) if minute else int(calendar.min_minute)
        var s = int(second.value()[]) if second else int(calendar.min_second)
        var ms = int(m_second.value()[]) if day else int(
            calendar.min_milisecond
        )
        self.m_second = calendar.m_seconds_since_epoch(y, mon, d, h, m, s, ms)
        self.hash = int(hash_val.value()[]) if hash_val else int(
            calendar.hash[_cal_h64](y, mon, d, h, m, s, ms)
        )

    @always_inline("nodebug")
    fn get_year(self) -> UInt64:
        """Get the year assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_y) >> _cal_h64.shift_64_y

    @always_inline("nodebug")
    fn get_month(self) -> UInt64:
        """Get the month assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_mon) >> _cal_h64.shift_64_mon

    @always_inline("nodebug")
    fn get_day(self) -> UInt64:
        """Get the day assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_d) >> _cal_h64.shift_64_d

    @always_inline("nodebug")
    fn get_hour(self) -> UInt64:
        """Get the hour assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_h) >> _cal_h64.shift_64_h

    @always_inline("nodebug")
    fn get_minute(self) -> UInt64:
        """Get the minute assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_m) >> _cal_h64.shift_64_m

    @always_inline("nodebug")
    fn get_second(self) -> UInt64:
        """Get the second assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_s) >> _cal_h64.shift_64_s

    @always_inline("nodebug")
    fn get_m_second(self) -> UInt64:
        """Get the m_second assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_ms) >> _cal_h64.shift_64_ms

    @always_inline("nodebug")
    fn get_attrs(
        self,
    ) -> (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64):
        """Get the year, month, day, hour, minute, second, milisecond
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (
            self.get_year(),
            self.get_month(),
            self.get_day(),
            self.get_hour(),
            self.get_minute(),
            self.get_second(),
            self.get_m_second(),
        )

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: Optional[Int] = None,
        owned month: Optional[Int] = None,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
        owned minute: Optional[Int] = None,
        owned second: Optional[Int] = None,
        owned m_second: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.

        Returns:
            Self.
        """
        var s = self
        if year:
            s.hash = (s.hash & ~_cal_h64.mask_64_y) | (
                year.unsafe_take() << _cal_h64.shift_64_y
            )
        if month:
            s.hash = (s.hash & ~_cal_h64.mask_64_mon) | (
                month.unsafe_take() << _cal_h64.shift_64_mon
            )
        if day:
            s.hash = (s.hash & ~_cal_h64.mask_64_d) | (
                day.unsafe_take() << _cal_h64.shift_64_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h64.mask_64_h) | (
                day.unsafe_take() << _cal_h64.shift_64_h
            )
        if minute:
            s.hash = (s.hash & ~_cal_h64.mask_64_m) | (
                day.unsafe_take() << _cal_h64.shift_64_m
            )
        if second:
            s.hash = (s.hash & ~_cal_h64.mask_64_s) | (
                day.unsafe_take() << _cal_h64.shift_64_s
            )
        if m_second:
            s.hash = (s.hash & ~_cal_h64.mask_64_ms) | (
                day.unsafe_take() << _cal_h64.shift_64_ms
            )
        return s

    @always_inline("nodebug")
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds since epoch.
        """
        return self.m_second // 1000

    @always_inline("nodebug")
    fn m_seconds_since_epoch(self) -> UInt64:
        """Miliseconds since the begining of the calendar's epoch.

        Returns:
            Miliseconds since epoch.
        """
        return self.m_second

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.m_second += other.m_second
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.m_second -= other.m_second
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.m_second += other.m_second

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.m_second -= other.m_second

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second == other.m_second

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second != other.m_second

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second > other.m_second

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second >= other.m_second

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second <= other.m_second

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second < other.m_second

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.m_second)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, seconds: Int = 0, m_seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            seconds: Seconds.
            m_seconds: Miliseconds.

        Returns:
            Self.
        """
        self.m_second += seconds * 1000 + m_seconds
        return self

    @always_inline("nodebug")
    fn subtract(owned self, seconds: Int = 0, m_seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            seconds: Seconds.
            m_seconds: Miliseconds.

        Returns:
            Self.
        """
        self.m_second -= seconds * 1000 + m_seconds
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime64` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime64().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime64` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        var ms = time.now() // 1_000_000
        var s = ms // 1_000
        return DateTime64.from_unix_epoch(s).add(m_seconds=ms)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var s = self.get_attrs()
        var time = dt_str.to_iso[iso](
            int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5])
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a dateTime64time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime64(
                p[0], p[1], p[2], p[3], p[4], p[5], calendar=calendar
            )
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt64, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime64` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h64](int(value))
        return DateTime64(
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            calendar=calendar,
            hash_val=value,
        )


@register_passable("trivial")
struct DateTime32(Hashable, Stringable):
    """Fast `DateTime32 ` struct. This is a "normal" `DateTime`
    with minute resolution. Uses given calendar's epoch
    and other params at build time. Assumes all instances have
    the same timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        - year: Up to year 4_096.
        - month: Up to month 16.
        - day: Up to day 32.
        - hour: Up to hour 32.
        - minute: Up to minute 64.

    - Minute Resolution (Gregorian as reference):
        - year: Up to year 8_171 since calendar epoch.

    - Notes:
        - Once methods that alter the underlying minutes
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var minute: UInt32
    """Minute."""
    var hash: UInt32
    """Hash."""

    fn __init__[
        T: _IntCollect = UInt16, A: _IntCollect = UInt8, B: _IntCollect = UInt32
    ](
        inout self,
        year: Optional[T] = None,
        month: Optional[A] = None,
        day: Optional[A] = None,
        hour: Optional[A] = None,
        minute: Optional[A] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[B] = None,
    ):
        """Construct a `DateTime32 ` from valid values.
        UTCCalendar is the default.

        Parameters:
            T: Any Intable Collectable type.
            A: Any Intable Collectable type.
            B: Any Intable Collectable type.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = int(year.value()[]) if year else int(calendar.min_year)
        var mon = int(month.value()[]) if month else int(calendar.min_month)
        var d = int(day.value()[]) if day else int(calendar.min_day)
        var h = int(hour.value()[]) if hour else int(calendar.min_hour)
        var m = int(minute.value()[]) if minute else int(calendar.min_minute)
        self.minute = (
            calendar.seconds_since_epoch(
                y, mon, d, h, m, int(calendar.min_second)
            )
            // 60
        ).cast[DType.uint32]()
        self.hash = int(hash_val.value()[]) if hash_val else int(
            calendar.hash[_cal_h32](y, mon, d, h, m)
        )

    @always_inline("nodebug")
    fn get_year(self) -> UInt32:
        """Get the year assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_y) >> _cal_h32.shift_32_y

    @always_inline("nodebug")
    fn get_month(self) -> UInt32:
        """Get the month assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_mon) >> _cal_h32.shift_32_mon

    @always_inline("nodebug")
    fn get_day(self) -> UInt32:
        """Get the day assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_d) >> _cal_h32.shift_32_d

    @always_inline("nodebug")
    fn get_hour(self) -> UInt32:
        """Get the hour assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_h) >> _cal_h32.shift_32_h

    @always_inline("nodebug")
    fn get_minute(self) -> UInt32:
        """Get the minute assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_m) >> _cal_h32.shift_32_m

    @always_inline("nodebug")
    fn get_attrs(
        self,
    ) -> (UInt32, UInt32, UInt32, UInt32, UInt32):
        """Get the year, month, day, hour, minute, second
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (
            self.get_year(),
            self.get_month(),
            self.get_day(),
            self.get_hour(),
            self.get_minute(),
        )

    @always_inline("nodebug")
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds since epoch.
        """
        return self.minute.cast[DType.uint64]() * 60

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: Optional[Int] = None,
        owned month: Optional[Int] = None,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
        owned minute: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.

        Returns:
            Self.
        """
        var s = self
        if year:
            s.hash = (s.hash & ~_cal_h32.mask_32_y) | (
                year.unsafe_take() << _cal_h32.shift_32_y
            )
        if month:
            s.hash = (s.hash & ~_cal_h32.mask_32_mon) | (
                month.unsafe_take() << _cal_h32.shift_32_mon
            )
        if day:
            s.hash = (s.hash & ~_cal_h32.mask_32_d) | (
                day.unsafe_take() << _cal_h32.shift_32_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h32.mask_32_h) | (
                day.unsafe_take() << _cal_h32.shift_32_h
            )
        if minute:
            s.hash = (s.hash & ~_cal_h32.mask_32_m) | (
                day.unsafe_take() << _cal_h32.shift_32_m
            )
        return s

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.minute += other.minute
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.minute -= other.minute
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.minute += other.minute

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.minute -= other.minute

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute == other.minute

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute != other.minute

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute > other.minute

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute >= other.minute

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute <= other.minute

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute < other.minute

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt32:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt32:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt32:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.minute)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, minutes: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            minutes: Minutes.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.minute += minutes + seconds * 60
        return self

    @always_inline("nodebug")
    fn subtract(owned self, minutes: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            minutes: Minutes.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.minute -= minutes + seconds * 60
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime32 ` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime32().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime32 ` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime32.from_unix_epoch(time.now() // 1_000_000_000)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso(
            int(self.get_year()),
            int(self.get_month()),
            int(self.get_day()),
            int(self.get_hour()),
            int(self.get_minute()),
            int(_calendar.min_second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a `DateTime32` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime32(p[0], p[1], p[2], p[3], p[4], calendar=calendar)
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt32, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime32 ` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h32](int(value))
        return DateTime32(
            d[0], d[1], d[2], d[3], d[4], calendar=calendar, hash_val=value
        )


@register_passable("trivial")
struct DateTime16(Hashable, Stringable):
    """Fast `DateTime16 ` struct. This is a `DateTime` with
    hour resolution, it can be used as a year, dayofyear,
    hour representation. uses given calendar's epoch and
    other params at build time. Assumes all instances have
    the same timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        year: Up to year 4.
        day: Up to day 512.
        hour: Up to hour 32.

    - Hour Resolution (Gregorian as reference):
        - year: Up to year 7 since calendar epoch.

    - Notes:
        - Once methods that alter the underlying m_seconds
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var hour: UInt16
    """Hour."""
    var hash: UInt16
    """Hash."""

    fn __init__[
        T: _IntCollect = UInt16, A: _IntCollect = UInt8
    ](
        inout self,
        year: Optional[T] = None,
        month: Optional[A] = None,
        day: Optional[A] = None,
        hour: Optional[A] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[T] = None,
    ):
        """Construct a `DateTime16` from valid values.
        UTCCalendar is the default.

        Parameters:
            T: Any Intable Collectable type.
            A: Any Intable Collectable type.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = int(year.value()[]) if year else int(calendar.min_year)
        var mon = int(month.value()[]) if month else int(calendar.min_month)
        var d = int(day.value()[]) if day else int(calendar.min_day)
        var h = int(hour.value()[]) if hour else int(calendar.min_hour)
        var m = int(calendar.min_minute)
        var s = int(calendar.min_second)
        self.hour = (
            calendar.seconds_since_epoch(y, mon, d, h, m, s) // (60 * 60)
        ).cast[DType.uint16]()
        self.hash = int(hash_val.value()[]) if hash_val else int(
            calendar.hash[_cal_h16](y, mon, d, h, m, s)
        )

    @always_inline("nodebug")
    fn get_year(self) -> UInt16:
        """Get the year assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h16.mask_16_y) >> _cal_h16.shift_16_y

    @always_inline("nodebug")
    fn get_day(self) -> UInt16:
        """Get the day assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h16.mask_16_d) >> _cal_h16.shift_16_d

    @always_inline("nodebug")
    fn get_hour(self) -> UInt16:
        """Get the hour assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h16.mask_16_h) >> _cal_h16.shift_16_h

    @always_inline("nodebug")
    fn get_attrs(self) -> (UInt16, UInt16, UInt16):
        """Get the year, month, day, hour, minute, second
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (self.get_year(), self.get_day(), self.get_hour())

    @always_inline
    fn replace(
        owned self,
        *,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            day: Day.
            hour: Hour.

        Returns:
            Self.
        """
        var s = self
        if day:
            s.hash = (s.hash & ~_cal_h16.mask_16_d) | (
                day.unsafe_take() << _cal_h16.shift_16_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h16.mask_16_h) | (
                day.unsafe_take() << _cal_h16.shift_16_h
            )
        return s

    @always_inline
    fn seconds_since_epoch(self) -> UInt32:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds.
        """
        return self.hour.cast[DType.uint32]() * 60 * 60

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour += other.hour
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour -= other.hour
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.hour += other.hour

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.hour -= other.hour

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour == other.hour

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour != other.hour

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour > other.hour

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour >= other.hour

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour <= other.hour

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour < other.hour

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt16:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt16:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt16:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.hour)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.hour += hours + seconds * 60 * 60
        return self

    @always_inline("nodebug")
    fn subtract(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.hour -= hours + seconds * 60 * 60
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime16 ` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime16().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime16 ` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime16.from_unix_epoch(time.now() // 1_000_000_000)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso[iso](
            int(self.get_year()),
            int(_calendar.min_month),
            int(self.get_day()),
            int(self.get_hour()),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a `DateTime16` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime16(p[0], p[1], p[2], p[3], calendar=calendar)
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt16, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime16 ` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h16](int(value))
        return DateTime16(
            year=d[0], day=d[2], hour=d[3], calendar=calendar, hash_val=value
        )


@register_passable("trivial")
struct DateTime8(Hashable, Stringable):
    """Fast `DateTime8 ` struct. This is a `DateTime`
    with hour resolution, it can be used as a dayofweek,
    hour representation. uses given calendar's epoch and
    other params at build time. Assumes all instances have
    the same timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        - day: Up to day 8.
        - hour: Up to hour 32.

    - Hour Resolution (Gregorian as reference):
        - hour: Up to hour 256 (~ 10 days) since calendar epoch.

    - Notes:
        - Once methods that alter the underlying m_seconds
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var hour: UInt8
    """Hour."""
    var hash: UInt8
    """Hash."""

    fn __init__[
        T: _IntCollect = Int, A: _IntCollect = Int
    ](
        inout self,
        year: Optional[T] = None,
        month: Optional[A] = None,
        day: Optional[A] = None,
        hour: Optional[A] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[A] = None,
    ):
        """Construct a `DateTime8 ` from valid values.
        UTCCalendar is the default.

        Parameters:
            T: Any Intable Collectable type.
            A: Any Intable Collectable type.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = int(year.value()[]) if year else int(calendar.min_year)
        var mon = int(month.value()[]) if month else int(calendar.min_month)
        var d = int(day.value()[]) if day else int(calendar.min_day)
        var h = int(hour.value()[]) if hour else int(calendar.min_hour)
        var m = int(calendar.min_minute)
        var s = int(calendar.min_second)
        self.hour = (
            calendar.seconds_since_epoch(y, mon, d, h, m, s) // (60 * 60)
        ).cast[DType.uint8]()
        self.hash = int(hash_val.value()[]) if hash_val else int(
            calendar.hash[_cal_h8](y, mon, d, h, m, s)
        )

    @always_inline("nodebug")
    fn get_day(self) -> UInt8:
        """Get the day assuming the hash is valid.

        Returns:
            The day.
        """
        return (self.hash & _cal_h8.mask_8_d) >> _cal_h8.shift_8_d

    @always_inline("nodebug")
    fn get_hour(self) -> UInt8:
        """Get the hour assuming the hash is valid.

        Returns:
            The hour.
        """
        return (self.hash & _cal_h8.mask_8_h) >> _cal_h8.shift_8_h

    @always_inline("nodebug")
    fn get_attrs(
        self,
    ) -> (UInt8, UInt8):
        """Get the year, month, day, hour, minute, second
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (self.get_day(), self.get_hour())

    @always_inline
    fn replace(
        owned self,
        *,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            day: Day.
            hour: Hour.

        Returns:
            Self.
        """
        var s = self
        if day:
            s.hash = (s.hash & ~_cal_h8.mask_8_d) | (
                day.unsafe_take() << _cal_h8.shift_8_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h8.mask_8_h) | (
                day.unsafe_take() << _cal_h8.shift_8_h
            )
        return s

    @always_inline("nodebug")
    fn seconds_since_epoch(self) -> UInt16:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds.
        """
        return self.hour.cast[DType.uint16]() * 60 * 60

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour += other.hour
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour -= other.hour
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.hour += other.hour

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.hour -= other.hour

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour == other.hour

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour != other.hour

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour > other.hour

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour >= other.hour

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour <= other.hour

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour < other.hour

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt8:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt8:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt8:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.hour)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.hour += hours + seconds * 60 * 60
        return self

    @always_inline("nodebug")
    fn subtract(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.hour -= hours + seconds * 60 * 60
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime8 ` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime8().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime8 ` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime8.from_unix_epoch(time.now() // 1_000_000_000)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso[iso](
            int(_calendar.min_year),
            int(_calendar.min_month),
            int(self.get_day()),
            int(self.get_hour()),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a `DateTime8` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime8(p[0], p[1], p[2], p[3], calendar=calendar)
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt8, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime8` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h8](int(value))
        return DateTime8(day=d[2], hour=d[3], calendar=calendar, hash_val=value)
