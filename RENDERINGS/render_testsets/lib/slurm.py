import math, re, time, subprocess, datetime

def sec2time(s):
    days = math.floor(s/86400)
    s -= days*86400
    hours = math.floor(s/3600)
    s -= hours*3600
    minutes = math.ceil(s/60)
    return f'{days}-{hours:02d}:{minutes:02d}'

def time2sec(t):
    x = re.split('-|:', t)
    x = [ int(k) for k in x ]
    if len(x)==1:
        x = [ 0, 0, x[0] ]
    elif len(x)==2:
        x = [ 0, x[0], x[1] ]
    return 86400*x[0] + 3600*x[1] + 60*x[2]

def waitjobs(jobs, verbose=True, sleeptime=300):
    if isinstance(jobs, int):
        joblist = [ jobs ]
    else:
        joblist = jobs.copy()
    time0 = time.time()
    while True:
        if verbose:
            subprocess.run('sq', shell=True)
        jobcopy = joblist.copy()
        for j in jobcopy:
            cmd = f'sq | grep {j} | wc -l'
            result = subprocess.run(cmd, capture_output=True, encoding='utf-8', shell=True)
            if int(result.stdout)==0:
                joblist.remove(j)
        if verbose:
            dt = math.floor(time.time()-time0)
            print(f'{len(joblist)} jobs remaining, elapsed time {datetime.timedelta(seconds=dt)}\n')
        if len(joblist)==0:
            break
        time.sleep(sleeptime)

