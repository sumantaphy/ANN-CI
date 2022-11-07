import sys

inputfile = sys.argv[1]

def readInput():
    fin = open(inputfile,"r")
    lines = fin.readlines()
    length = len(lines)
    for i in range(length):
        toks = lines[i].split(",")
        if len(toks) >= 2:
            if toks[0] == 'test_fraction':
                fr_test = float(toks[1])

            if toks[0] == 'inp_dim':
                n_inputs = int(toks[1])

            if toks[0] == 'node_amplitude':
                H = int(toks[1])

            if toks[0] == 'node_sign':
                H_s = int(toks[1])

            if toks[0] == 'learn_rate':
                l_rate = float(toks[1])

            if toks[0] == 'batch_size':
                batch = int(toks[1])

            if toks[0] == 'num_epochs_amplitude':
                epoch_a = int(toks[1])

            if toks[0] == 'num_epochs_sign':
                epoch_s = int(toks[1])

            if toks[0] == 'input_file':
                input_file = str(toks[1]).strip()

            if toks[0] == 'csv_column':
                start = int(toks[1])
                end = int(toks[2])
                ci = int(toks[3])
                det = int(toks[4])
                det_sign = int(toks[5])

    inp_list = []
    for i in range(start,end+1):
        inp_list.append(i)
    inp_list.append(ci)

    return fr_test,n_inputs,H, H_s, batch, l_rate, input_file, inp_list, det, det_sign, epoch_a, epoch_s
