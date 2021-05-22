import argparse
import os


def argparser():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument(
        '--window_lengths',
        type=int,
        nargs='+',
        help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
    )
    parser.add_argument(
        '--num_windows',
        type=int,
        nargs='+',
        help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)'
    )
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=1000,
        help='Number of neurons in hidden layer.'
    )
    parser.add_argument(
        '--regularizer',
        type=float,
        default=0.001,
        help='(Lambda value / 2) of L2 regularizer on weights connected to last layer (0 to exclude).'
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.7,
        help='Rate to be kept for dropout.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Number of classes (families).'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=22152,
        help='Length of input sequences.'
    )
    # for learning
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=100,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default='E:/Tarn-master thesis/SAMPLE-PSSM-ionchannel/ionchannels/trn_CH_LS_SM.pkl',
        help='Directory for input data.'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='E:/Tarn-master thesis/SAMPLE-PSSM-ionchannel/ionchannels/tst_CH_LS_SM.pkl',
        help='Directory for input data.'
    )
    parser.add_argument(
        '--prev_checkpoint_path',
        type=str,
        default='',
        help='Restore from pre-trained model if specified.'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Path to write checkpoint file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Directory for log data.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Interval of steps for logging.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=100,
        help='Interval of steps for save model.'
    )
    # test
    parser.add_argument(
        '--fine_tuning',
        type=bool,
        default=False,
        help='If true, weight on last layer will not be restored.'
    )
    parser.add_argument(
        '--fine_tuning_layers',
        type=str,
        nargs='+',
        default=["fc2"],
        help='Which layers should be restored. Default is ["fc2"].'
    )
    parser.add_argument(
        '--out_file',
        type=str,
        default='./logs/outfile/raw_output.txt',
        help='File to print testing outbut. Default is "output.txt"'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold in Confusion Matrix. Default is 0.5'
    )
    parser.add_argument(
        '--num_feature',
        type=int,
        default=20,
        help='Number of features. If you use raw sequence, default value is 20.'
    )
    parser.add_argument(
        '--is_raw',
        type=lambda x: (str(x).lower() not in ['false', '0', 'no']),
        default=True,
        help='Train raw sequence.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    # check validity
    # assert (len(FLAGS.window_lengths) == len(FLAGS.num_windows))

    return FLAGS


def logging(msg, FLAGS):
    fpath = os.path.join(FLAGS.log_dir, "log.txt")
    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print(msg)
