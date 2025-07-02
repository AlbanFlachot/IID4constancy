import sys, os, subprocess, argparse, time
import numpy as np
import util, slurm

def task(cmd, jobnum=False, timeout=None, run=True):
    print(cmd)
    if run:
        result = subprocess.run(cmd, timeout=300, capture_output=True, encoding='utf-8', shell=True)
        print(result.stdout)
        if jobnum:
            job = int(result.stdout.split()[3])
        else:
            job = result.stdout
    else:
        job = 0
    return job

class RenderJob:

    def __init__(self, imagecmd, ntrain, nval, ntest, nbatch=10, imagetime=10, posttime=300, filetag='render', sleeptime=30, email=''):
        self.imagecmd = imagecmd
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest
        self.nimage = ntrain + nval + ntest
        self.nbatch = nbatch
        self.imagetime = imagetime
        self.posttime = posttime
        self.filetag = filetag
        self.sleeptime = sleeptime
        self.emailopt = f'--mail-user={email} --mail-type=ALL' if email else ''
        self.batchfiles = []

    def render(self, batchlist=None):

        if os.path.exists(self.filetag):
            task(f'rm -fr {self.filetag}')
        os.mkdir(self.filetag)
        
        imagek = np.linspace(1, self.nimage+1, self.nbatch+1, dtype=np.int64)
        batchiter = [ k-1 for k in batchlist ] if batchlist else range(imagek.size-1)

        batches = []
        for k in batchiter:
        
            i1 = imagek[k]
            i2 = imagek[k+1]-1
            batchfile = f'{self.filetag}/{self.filetag}-{k+1:02d}.tar'
            print('imagecomd is:')
            print(self.imagecmd)
            batch = RenderBatch(imagecmd=self.imagecmd, low=i1, high=i2, batchfile=batchfile, imagetime=self.imagetime, posttime=self.posttime, emailopt=self.emailopt)
            batch.start()
            batches.append(batch)

        self.batchfiles = []
        while batches:
            time.sleep(self.sleeptime)
            print('checking batches...')
            sqstr = task('sq')
            rmlist = []
            for b in batches:
                if not b.inprogress(sqstr=sqstr):
                    if b.fileexists():
                        print(f'completed: {b.batchfile}')
                        rmlist.append(b)
                        self.batchfiles.append(b.batchfile)
                    else:
                        print(f'restarting: {b.batchfile}')
                        b.start()
            for b in rmlist:
                batches.remove(b)
            print([ b.batchfile for b in batches ])
    
    def tvt(self):
        sbatchcmd = f'sbatch --time=0-00:20 {self.emailopt} $SLURM_LIBDIR/job_tvt.sh {self.filetag}.tar {self.ntrain+1} {self.ntrain+self.nval} {self.ntrain+self.nval+1} {self.ntrain+self.nval+self.ntest} \'{" ".join(self.batchfiles)}\''
        jobnum = task(sbatchcmd, jobnum=True)
        slurm.waitjobs(jobnum, sleeptime=self.sleeptime)
        task(f'rm -fr {self.filetag}')

class RenderBatch:

    def __init__(self, imagecmd, low, high, batchfile, imagetime=10, posttime=300, emailopt=''):
        self.imagecmd = imagecmd
        self.low = low
        self.high = high
        self.batchfile = batchfile
        self.imagetime = imagetime
        self.rendertime = ((high-low)+1)*imagetime
        self.posttime = posttime
        self.totaltime = self.rendertime + self.posttime
        self.emailopt = emailopt
        self.jobnum = None

    def start(self):
        rendercmd = f'$SLURM_LIBDIR/render_util.py batch --low {self.low} --high {self.high} --imagedir $IMAGEDIR --batchfile {self.batchfile} --rendertime {self.rendertime} -- {self.imagecmd}'
        sbatchcmd = f'sbatch --time={slurm.sec2time(self.totaltime)} {self.emailopt} $SLURM_LIBDIR/job_sbatch.sh \'{rendercmd}\''
        print('sbatchcmbd is:')
        print(sbatchcmd)
        self.jobnum = task(sbatchcmd, timeout=self.totaltime, jobnum=True)

    def inprogress(self, sqstr=None):
        if self.jobnum is None:
            return False
        if sqstr is None:
            sqstr = task('sq')
        return sqstr.find(str(self.jobnum)) >= 0

    def fileexists(self):
        return os.path.exists(self.batchfile)

def batch(imagecmd, imagedir, low, high, batchfile, imagetime=10, rendertime=np.inf):

    if isinstance(imagecmd, str):
        imagecmd = imagecmd.split()
    rendermain = imagecmd[0]
    renderargs = ' '.join(imagecmd[1:])

    t0 = time.time()
    logfile = f'log_{low:06d}_{high:06d}.txt'
    def log(i, t=None):
        with open(logfile,'a') as f:
            if isinstance(i, str):
                f.write(i+'\n')
            else:
                f.write(f'{i}, {t:.2f}\n')

    log(f'{rendermain} {renderargs}; {low} to {high}')

    for i in range(low, high+1):
    
        filename = os.path.join(util.fullpath(imagedir), f'image{i:06d}')

        timeouts = 0
        while True:
            t1 = time.time()
            try:
                cmd = f'blender --background --python-use-system-env --python {rendermain} -- --filename {filename} {renderargs}'
                task(cmd, timeout=imagetime)
                log(i, imagetime)
                log(i, time.time()-t1)
                break
            except subprocess.TimeoutExpired:
                log(i, time.time()-t1)
                timeouts += 1
                if timeouts >= 3: break
    
        if timeouts >= 3: 
            log('We got a problem')
            break

        if time.time()-t0 > rendertime:
            log('batch timeout')
            with open('BATCH_ERRORS.txt','a') as f:
                f.write(f'batch {low} to {high} only rendered to {i}\n')
            break

    if not timeouts >= 3:
        dst = os.path.join(os.getcwd(), batchfile)
        os.chdir(imagedir)
        print(dst)
        task(f'ls {dst}')
        task(f'tar czf {dst} *')

def tvt(srcfiles, imagedir, dstfile, ntrain, nval, ntest):

    tmpdir = os.path.join(imagedir, 'tmp')
    os.mkdir(tmpdir)
    for fname in srcfiles:
        task(f'tar xf {fname} -C {tmpdir}')

    def moveto(subdir, i1, i2):
        fulldir = os.path.join(imagedir, subdir)
        os.mkdir(fulldir)
        filt = os.path.join(tmpdir, f'image{{{i1:06d}..{i2:06d}}}*')
        task(f'mv {filt} {fulldir}')

    if ntrain>0: moveto('train', 1, ntrain)
    if nval>0:   moveto('val', ntrain+1, ntrain+nval)
    if ntest>0:  moveto('test', ntrain+nval+1, ntrain+nval+ntest)
    task(f'rm -fr {tmpdir}')

    basedir = os.getcwd()
    dst = os.path.join(basedir, dstfile)
    os.chdir(imagedir)
    task(f'tar cf {dst} *')
    os.chdir(basedir)
    # task(f'rm {" ".join(srcfiles)}')

if __name__ == '__main__':

    cmd = sys.argv[1]
    argv1 = sys.argv[2:sys.argv.index('--')]
    argv2 = sys.argv[sys.argv.index('--')+1:]

    if cmd == 'batch':
        parser = argparse.ArgumentParser()
        parser.add_argument('--imagedir', type=str)
        parser.add_argument('--low', type=int)
        parser.add_argument('--high', type=int)
        parser.add_argument('--batchfile', type=str)
        parser.add_argument('--imagetime', type=float, default=10)
        parser.add_argument('--rendertime', type=float, default=np.inf)
        args = parser.parse_args(argv1)
        batch(imagecmd=argv2, imagedir=args.imagedir, low=args.low, high=args.high,
            batchfile=args.batchfile, imagetime=args.imagetime, rendertime=args.rendertime)

    elif cmd == 'tvt':
        parser = argparse.ArgumentParser()
        parser.add_argument('--imagedir', type=str, default='images')
        parser.add_argument('--dstfile', type=str)
        parser.add_argument('--ntrain', type=int)
        parser.add_argument('--nval', type=int, default=0)
        parser.add_argument('--ntest', type=int, default=0)
        args = parser.parse_args(argv1)
        tvt(srcfiles=argv2, imagedir=args.imagedir, dstfile=args.dstfile,
            ntrain=args.ntrain, nval=args.nval, ntest=args.ntest)

    else:
        raise ValueError(f'unknown function {cmd}')

