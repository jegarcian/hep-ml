import os, sys
import csv
import random
import shutil

def main():
    
    from optparse import OptionParser     
    parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
    parser.add_option("-d","--dir",       dest="indir",           help="directory containing images (default: %default)")
    parser.add_option("-o","--outdir",    dest="outdir",          help="Output directory for csv (default: %default)")
    parser.add_option("-r","--ratio",     dest="ratio",           help="ratio for training (default: %default)")
    parser.add_option("-t","--total",     dest="total",           help="total images per category (default: %default)")
    parser.set_defaults(indir="DarkMachines/images", outdir="../InputImages", ratio="0.7,0.1,0.2", total=74000)
    (options,args) = parser.parse_args()

    bkg_types = {'Med2000' : 0, 'tchannel' : 1, 'Wjets' : 2}
    rbkg_types = { 0 : 'Med2000', 1 : 'tchannel', 2 : 'Wjets'}
    bkg_images = {'Med2000' : 0, 'tchannel' : 1, 'Wjets' : 2}

    f = []
    for (dirpath, dirnames, filenames) in os.walk(options.indir):
        for ifile in filenames :
            if "jpg" in ifile :
                for key in bkg_types :
                    nfile = os.path.join(dirpath, ifile)
                    if key in nfile and  bkg_images[key] < options.total :
                        f.append([nfile,bkg_types[key]])
                        #print([nfile,bkg_types[key]])
                        bkg_images[key] += 1


    
    rtrain, rval, rtest = [int(len(f)*float(x)) for x in options.ratio.split(",")]

#    k = int(len(f)*options.ratio)
    idx = random.sample(xrange(len(f)), rtrain)
    #print("idx",idx)

    nlist = [x for x in range(0,len(f))]
    random.shuffle(nlist) 

    print " Creating train dataset with ",rtrain, " images\n"
    output_train = os.path.join(options.outdir,'train')
    if not os.path.exists(output_train):
        os.makedirs(output_train)
    for x in bkg_types :
        type_train = os.path.join(output_train,x)
        if not os.path.exists(type_train):
            os.makedirs(type_train)


    with open(output_train + '/train.csv', mode='w') as train_file:
        wtrain = csv.writer(train_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for ix in idx :
            wtrain.writerow(f[ix])
            path = os.path.join(output_train,rbkg_types[f[ix][1]])
            shutil.copy(f[ix][0],path)
            #print(f[ix][0],path)
            f[ix] = 0


    print " Creating validation dataset with ",rval, " images\n"
    output_val = os.path.join(options.outdir,'val')
    if not os.path.exists(output_val):
        os.makedirs(output_val)
    for x in bkg_types :
        type_val = os.path.join(output_val,x)
        if not os.path.exists(type_val):
            os.makedirs(type_val)


    ival = 0
    with open(output_val + '/val.csv', mode='w') as val_file:
        wval = csv.writer(val_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for ix in nlist :
            if f[ix] != 0 and ival < rval :
                wval.writerow(f[ix])
                path = os.path.join(output_val,rbkg_types[f[ix][1]])
                shutil.copy(f[ix][0],path)
                ival = ival + 1
                #print(f[ix][0],path)
                f[ix] = 0


    print " Creating validation dataset with ",rtest, " images\n"
    output_test = os.path.join(options.outdir,'test')
    if not os.path.exists(output_test):
        os.makedirs(output_test)
    for x in bkg_types :
        type_test = os.path.join(output_test,x)
        if not os.path.exists(type_test):
            os.makedirs(type_test)


    with open(output_test + '/test.csv', mode='w') as test_file:
        wtest = csv.writer(test_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for ix in range(0,len(f)) :
            if f[ix] != 0 :
                wtest.writerow(f[ix])
                path = os.path.join(output_test,rbkg_types[f[ix][1]])
                shutil.copy(f[ix][0],path)
                #print(f[ix][0],path)


# --------------------------------------------------------------- 
# Main 
# ---------------------------------------------------------------           
if __name__ == '__main__':
    main()
