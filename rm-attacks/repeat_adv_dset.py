from datasets import Dataset
import pandas as pd
import argparse
from make_adv_dset import combresp, make_rmset
    
def main(args):
    MAXROUNDS = 5
    ROUNDCANDS = 10
    SDEVICE=0
    
    rmpath = args.rmpath
    # load reward model (sanity check RM)
    print("loading dataset")
    startdf = pd.read_json(args.inpname, orient='records', lines=True)
    print(len(startdf))
    # get starter dataset (new data) to mess around with while using the RM
    startdset = Dataset.from_pandas(startdf)
    startdset = startdset.shuffle(seed=0)
    
    print("mapping")
    # HACK note that we're taking j from the first set so that we start with the orig
    # take random strings from either chosen or rejected
    startdset = startdset.map(combresp)
    startdset = startdset.filter(lambda x: len(x["instr"]) < 4000, batched=False)
    
    # generate adversarial dataset, store automatically
    make_rmset(rmpath, startdset, MAXROUNDS, ROUNDCANDS, SDEVICE, args.fname)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='My Python script.')
    parser.add_argument('rmpath', type=str, help='file to read in stuff from')
    parser.add_argument('inpname', type=str, help='file to read in stuff from')
    parser.add_argument('fname', type=str, help='file to output stuff to')

    progargs = parser.parse_args()
    main(progargs)