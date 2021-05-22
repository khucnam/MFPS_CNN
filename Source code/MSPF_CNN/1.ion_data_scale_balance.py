import os
import pandas as pd
import numpy as np
import time
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

os.chdir("E:/Tarn-master thesis/PSSM-ionchannel")


class BasicFunctions:
    '''
        Constructor
    '''

    def __init__(self):
        print("Start your Class...")

    def linearScale(self, inputs):
        minData = np.min(inputs)
        maxData = np.max(inputs)
        scores = [round((x - minData) / (maxData - minData), 6) for x in inputs]

        return scores

    def zScoreScale(self, inputs):
        meanData = np.mean(inputs)
        stdData = np.std(inputs, ddof=1)
        scores = [round(((x - meanData) / stdData), 6) for x in inputs]

        return scores

    def readPSSMFile(self, inputFilePath, SeqName=False, SelfStore=False):
        # A technque to read PSSM profiles using panda
        # Start index is in col 2 to get all content only.
        # if set to 1 means select with their protein amino acid in first col.
        startCol = 2
        if SeqName == True:
            startCol = 1
            # 22 amino acid matrix, 42 means with their frequence
        endCol = 22
        # Set some of unuseful rows by number of row
        setSkipRows = (0, 1, 2)
        # Read data
        pssmData = pd.read_csv(inputFilePath,
                               skiprows=setSkipRows,
                               # White space delimater ignores more spaces
                               delim_whitespace=True,
                               # No header
                               header=None,
                               usecols=range(startCol, endCol)
                               )
        # Remove last 5 rows
        maxRow = (len(pssmData.index)) - 6
        # Select only content of PSSM profiles
        pssmData = pssmData.loc[0:maxRow, :]
        #         print(pssmData)

        if SelfStore == True:
            self.pssm = pssmData


        else:
            print(inputFilePath," pssmData.shape ",pssmData.shape)
            return pssmData

    def pssmDefaultFeatureGenerator(self, inputDir, Scale=None):
        # Set amino acid
        # AMINO_ACID = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
        #               'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        # Create new dataframe to store
        COLUMN_NAMES = ['Data']
        COLLECT = pd.DataFrame(columns=COLUMN_NAMES)

        # Set your time
        start_time = time.clock()
        max_length = 0
        #         result = []
        # Loop all files
        for root, dirs, files in os.walk(inputDir):
            count = 0
            i = 0
            p = 0
            for file in files:
                filePath = os.path.join(root, file)
                # print("File {0}: {1}".format(i+1, filePath));
                # Read file pssm profile
                self.readPSSMFile(filePath, SeqName=True, SelfStore=True)
                # Get Sequence Lenghth
                alldata = self.pssm.iloc[:, 1:21]
                alldata = np.array(alldata).flatten()
                SEQ_LENGTH = len(self.pssm.index)
                result = []
                # print(len(alldata))
                if Scale == 'linearscale':
                    result.append(self.linearScale(alldata))
                elif Scale == "zscorescale":
                    result.append(self.zScoreScale(alldata))

                elif Scale == "none":
                    result.append(alldata)

                if max_length < SEQ_LENGTH:
                    max_length = SEQ_LENGTH

                # Insert to data frame
                COLLECT.loc[i] = [result]
                # increase
                i = i + 1
                p += 1

            # Print total file
            print("--Time: {0} seconds, Total PSSM files: {1}--".format(round((time.clock() - start_time), 4), i))
            # print('seq above 4k : %d' % count)

        return pd.DataFrame(COLLECT['Data'].values.tolist()), max_length


if __name__ == '__main__':
    # data preprocessing
    x = BasicFunctions()
    # set data directions
    WorkDir = "E:/Tarn-master thesis/SAMPLE-PSSM-ionchannel"

    trnMem = WorkDir + "/membraneproteins/train/"
    testMem = WorkDir + "/membraneproteins/test/"

    trnTrans = WorkDir + "/iontransporters/train/"
    testTrans = WorkDir + "/iontransporters/test/"

    trnChans = WorkDir + "/ionchannels/train/"
    tesChans = WorkDir + "/ionchannels/test/"

    # set scaling
    # setScaling = 'none'
    setScaling = 'linearscale'
    # setScaling = 'zscorescale'
    print("read data : Start...")
    trn_mem, max_trn_tm = x.pssmDefaultFeatureGenerator(trnMem, Scale=setScaling)
    trn_tt, max_trn_tt = x.pssmDefaultFeatureGenerator(trnTrans, Scale=setScaling)
    trn_ch, max_trn_tc = x.pssmDefaultFeatureGenerator(trnChans, Scale=setScaling)

    tst_mem, max_tst_tm = x.pssmDefaultFeatureGenerator(testMem, Scale=setScaling)
    tst_tt, max_tst_tt = x.pssmDefaultFeatureGenerator(testTrans, Scale=setScaling)
    tst_ch, max_tst_tc = x.pssmDefaultFeatureGenerator(tesChans, Scale=setScaling)

    print("read data : End")
    # print(trn_ch_long)

    # check the max sequence
    print("max sequence check :")
    print(max_trn_tm, max_trn_tt, max_trn_tc)
    print(max_tst_tm, max_tst_tt, max_tst_tc)

    seqlen = max(max_trn_tm, max_trn_tt, max_trn_tc, max_tst_tm, max_tst_tt, max_tst_tc)

    trn_ch['class'] = 1
    tst_ch['class'] = 1
    trn_tt['class'] = 0
    tst_tt['class'] = 0
    trn_mem['class'] = 0
    tst_mem['class'] = 0

    print("set channel Train and Test data...")
    # Train
    trainChannelMem = pd.DataFrame()
    trainChannelMem = trainChannelMem.append(trn_ch).append(trn_tt).append(trn_mem)
    # Test
    testChannelMem = pd.DataFrame()
    testChannelMem = testChannelMem.append(tst_ch).append(tst_tt).append(tst_mem)

    # set iontransporter as pos
    trn_ch['class'] = 0
    tst_ch['class'] = 0
    trn_tt['class'] = 1
    tst_tt['class'] = 1
    trn_mem['class'] = 0
    tst_mem['class'] = 0

    print("set transporter Train and Test data...")
    # Train
    trainTransportersMem = pd.DataFrame()
    trainTransportersMem = trainTransportersMem.append(trn_tt).append(trn_ch).append(trn_mem)
    # Test
    testTransportersMem = pd.DataFrame()
    testTransportersMem = testTransportersMem.append(tst_tt).append(tst_ch).append(tst_mem)
    print(trainTransportersMem)

    # set membrane as pos
    trn_ch['class'] = 0
    tst_ch['class'] = 0
    trn_tt['class'] = 0
    tst_tt['class'] = 0
    trn_mem['class'] = 1
    tst_mem['class'] = 1

    print("set membrane Train and Test data...")
    # Train
    trainChannelTrans = pd.DataFrame()
    trainChannelTrans = trainChannelTrans.append(trn_mem).append(trn_tt).append(trn_ch)
    # Test
    testChannelTrans = pd.DataFrame()
    testChannelTrans = testChannelTrans.append(tst_mem).append(tst_tt).append(tst_ch)
    print(trainChannelTrans)

    # channel data
    x_train_init_ch = trainChannelMem.iloc[:, :-1]
    y_train_init_ch = trainChannelMem.iloc[:, -1]
    x_test_init_ch = testChannelMem.iloc[:, :-1]
    y_test_init_ch = testChannelMem.iloc[:, -1]

    # transporter data
    x_train_init_tt = trainTransportersMem.iloc[:, :-1]
    y_train_init_tt = trainTransportersMem.iloc[:, -1]
    x_test_init_tt = testTransportersMem.iloc[:, :-1]
    y_test_init_tt = testTransportersMem.iloc[:, -1]

    # membrane data
    x_train_init_mem = trainChannelTrans.iloc[:, :-1]
    y_train_init_mem = trainChannelTrans.iloc[:, -1]
    x_test_init_mem = testChannelTrans.iloc[:, :-1]
    y_test_init_mem = testChannelTrans.iloc[:, -1]


    def padzero(arr, max_len):
        gen = np.zeros((len(arr), (max_len * 20)))
        for i, data in enumerate(gen):
            data[0:len(arr[i][0])] = arr[i][0]
        return gen


    pad_trn_ch = padzero(x_train_init_ch.values, seqlen)
    pad_trn_tt = padzero(x_train_init_tt.values, seqlen)
    pad_trn_mem = padzero(x_train_init_mem.values, seqlen)

    pad_tst_ch = padzero(x_test_init_ch.values, seqlen)
    pad_tst_tt = padzero(x_test_init_tt.values, seqlen)
    pad_tst_mem = padzero(x_test_init_mem.values, seqlen)

    print(pad_trn_ch.shape)

    # imbalanced data handle
    print("imbalanced handling start:...")
    setsample = 'ramdomoversampler'
    # setsample = 'adasyn'
    # setsample = 'smote'
    if setsample == 'ramdomoversampler':
        sample = RandomOverSampler()
        x_train_balanced_ch, y_train_balanced_ch = sample.fit_resample(pad_trn_ch, y_train_init_ch.values)
        x_train_balanced_tt, y_train_balanced_tt = sample.fit_resample(pad_trn_tt, y_train_init_tt.values)
        x_train_balanced_mem, y_train_balanced_mem = sample.fit_resample(pad_trn_mem, y_train_init_mem.values)
    elif setsample == 'smote':
        sample = SMOTE()
        x_train_balanced_ch, y_train_balanced_ch = sample.fit_resample(pad_trn_ch, y_train_init_ch.values)
        x_train_balanced_tt, y_train_balanced_tt = sample.fit_resample(pad_trn_tt, y_train_init_tt.values)
        x_train_balanced_mem, y_train_balanced_mem = sample.fit_resample(pad_trn_mem, y_train_init_mem.values)
    elif setsample == 'adasyn':
        sample = ADASYN()
        x_train_balanced_ch, y_train_balanced_ch = sample.fit_resample(pad_trn_ch, y_train_init_ch.values)
        x_train_balanced_tt, y_train_balanced_tt = sample.fit_resample(pad_trn_tt, y_train_init_tt.values)
        x_train_balanced_mem, y_train_balanced_mem = sample.fit_resample(pad_trn_mem, y_train_init_mem.values)
    print("imbalanced handling end")

    trn_ch_balanced = pd.DataFrame(x_train_balanced_ch)
    trn_ch_balanced['class'] = y_train_balanced_ch
    tst_ch_init = pd.DataFrame(pad_tst_ch)
    tst_ch_init['class'] = y_test_init_ch.values

    trn_tt_balanced = pd.DataFrame(x_train_balanced_tt)
    trn_tt_balanced['class'] = y_train_balanced_tt
    tst_tt_init = pd.DataFrame(pad_tst_tt)
    tst_tt_init['class'] = y_test_init_tt.values

    trn_mem_balanced = pd.DataFrame(x_train_balanced_mem)
    trn_mem_balanced['class'] = y_train_balanced_mem
    tst_mem_init = pd.DataFrame(pad_tst_mem)
    tst_mem_init['class'] = y_test_init_mem.values

    print(trn_ch_balanced)
    print(trn_tt_balanced)
    print(trn_mem_balanced)

    print("output data start")
    # channel data
    outputfile = "E:/Tarn-master thesis/PSSM-ionchannel/ionchannels/trn_CH_LS_SM.pkl"
    trn_ch_balanced.to_pickle(outputfile)
    outputfile = "E:/Tarn-master thesis/PSSM-ionchannel/ionchannels/tst_CH_LS_SM.pkl"
    tst_ch_init.to_pickle(outputfile)

    # transporter
    outputfile = "E:/Tarn-master thesis/PSSM-ionchannel/iontransporters/trn_TT_NS_ADA.pkl"
    trn_tt_balanced.to_pickle(outputfile)
    outputfile = "E:/Tarn-master thesis/PSSM-ionchannel/iontransporters/tst_TT_NS_ADA.pkl"
    tst_tt_init.to_pickle(outputfile)

    # membrane
    outputfile = "E:/Tarn-master thesis/PSSM-ionchannel/membraneproteins/trn_mem_NS_ADA.pkl"
    trn_mem_balanced.to_pickle(outputfile)
    outputfile = "E:/Tarn-master thesis/PSSM-ionchannel/membraneproteins/tst_mem_NS_ADA.pkl"
    tst_mem_init.to_pickle(outputfile)
    print("output data END")
