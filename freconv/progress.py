import os
import shutil
import sys
import time

__all__ = ['progress_bar']

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def _terminal_width():
    return shutil.get_terminal_size(fallback=(80, 20)).columns


def _non_tty_interval():
    try:
        return int(os.environ.get("FRE_PROGRESS_LOG_INTERVAL", "10"))
    except ValueError:
        return 10


def _should_log_non_tty(current, total):
    interval = _non_tty_interval()
    step = current + 1
    return current == 0 or step == total or (interval > 0 and step % interval == 0)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    total = max(total, 1)
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('Step: %s' % format_time(step_time))
    L.append('Tot: %s' % format_time(tot_time))
    if msg:
        L.append(msg)

    msg = ' | '.join(L)

    if not sys.stdout.isatty():
        if _should_log_non_tty(current, total):
            sys.stdout.write('Progress: %d/%d | %s\n' % (current + 1, total, msg))
            sys.stdout.flush()
        return

    term_width = _terminal_width()
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    msg = '  ' + msg
    sys.stdout.write(msg)
    for i in range(max(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3, 0)):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (hours, minutes, seconds)
