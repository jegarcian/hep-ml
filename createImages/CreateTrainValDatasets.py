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
    parser.set_defaults(indir="images", outdir="../InputImages", ratio=0.9, total=74000)
    (options,args) = parser.parse_args()

    bkg_types = {'ttbar' : 0, 'Wjets' : 1}
    rbkg_types = { 0 : 'ttbar',  1 : 'Wjets'}
    bkg_images = {'ttbar' : 0, 'Wjets' : 0}

    f = []
    for (dirpath, dirnames, filenames) in os.walk(options.indir):
        for ifile in filenames :
            if "jpg" in ifile and not 'Zjets' in dirpath :
                for key in bkg_types :
                    nfile = os.path.join(dirpath, ifile)
                    if key in nfile and  bkg_images[key] < options.total :
                        f.append([nfile,bkg_types[key]])
                        #print([nfile,bkg_types[key]])
                        bkg_images[key] += 1


    k = int(len(f)*options.ratio)
    idx = random.sample(xrange(len(f)), k)
    #print("idx",idx)


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


    output_val = os.path.join(options.outdir,'val')
    if not os.path.exists(output_val):
        os.makedirs(output_val)
    for x in bkg_types :
        type_val = os.path.join(output_val,x)
        if not os.path.exists(type_val):
            os.makedirs(type_val)


    with open(output_val + '/val.csv', mode='w') as val_file:
        wval = csv.writer(val_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for ix in range(0,len(f)) :
            if f[ix] != 0 :
                wval.writerow(f[ix])
                path = os.path.join(output_val,rbkg_types[f[ix][1]])
                shutil.copy(f[ix][0],path)
                #print(f[ix][0],path)

#    with open('employee_file.csv', mode='w') as employee_file:
#        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#        employee_writer.writerow(['John Smith', 'Accounting', 'November'])

# --------------------------------------------------------------- 
# Main 
# ---------------------------------------------------------------           
if __name__ == '__main__':
    main()
