import datetime

the_end = 2147483647000.0

day = 90000000.0
hour = 3600000.0
minute = 60000.0
second = 1000.0

days = int(the_end / day)
rest = the_end - (days * int(day))
hours = int(rest / hour)
rest = rest - (hours * int(hour))
minutes = int(rest / minute)
rest = rest - (int(minutes) * int(minute))
seconds = rest / second

d = datetime.datetime(1970, 1, 1)
d = d + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
print d.isoformat()


