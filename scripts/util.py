import os
import errno

def safe_mkdir(dir_path):

    try:
        os.makedirs(dir_path)
    except OSError, e:
        if e.errno != errno.EEXIST:
            print "Error creating directory %s" % dir_path
            sys.exit(2)
