import os
from vera_common import Commons
from glob import glob
from os import path
import sys

def run():
    ghostscriptPath = Commons.getEnv('GS_PATH')
    dirs = sys.argv[1:]
    gs_exe = path.join(ghostscriptPath, 'gswin64c.exe')
    for p in dirs:
        if not path.exists(p):
            print 'Error - path', p, ' does not exist. Skipping...'
            continue
        print 'Processing dir', p
        for pdf_path in glob(path.join(p, '*.pdf')) :
            head, tail = path.split(pdf_path)
            tiff = tail.replace('.pdf', '-%04d.tiff')
            tiff_path = path.join(head, tiff)
            print 'Convert', pdf_path, 'to TIFF', tiff_path
            if path.exists(tiff_path):
                os.unlink(tiff_path)
            #@TODO: PAPERSIZE has to be the same as the source!!!
            os.popen(' '.join([
                               gs_exe,
                               '-q',
                               '-dNOPAUSE',
                               '-dBATCH',
                               '-r300',
                               '-sDEVICE=tiffgray',
                               '-sPAPERSIZE=a4',
                               '-sOutputFile=%s %s' % (tiff_path, pdf_path),
                               ]))

if __name__ == '__main__':
    run()