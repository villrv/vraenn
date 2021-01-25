import numpy as np
from .lc import LightCurve
from .vraenn import prep_input, get_decoder, get_decodings
import argparse
from keras.models import model_from_json, Model
from keras.layers import Input
import datetime
import os

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))


def str2bool(v):
    """
    Helper function to turn strings to bool

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def feat_from_raenn(data_file, model_base=None,
                    prep_file=None, plot=False):
    """
    Calculate RAENN features

    Parameters
    ----------
    data_file : str
        Name of data file with light curves
    model_base : str
        Name of RAENN model file
    prep_file : str
        Name of file which encodes the feature prep

    Returns
    -------
    encodings : numpy.ndarray
        Array of object IDs (strings)

    TODO
    ----
    - prep file seems unnecessary
    """
    sequence, outseq, ids, maxlen, nfilts = prep_input(data_file, load=True, prep_file=prep_file)
    model_file = model_base + '.json'
    model_weight_file = model_base+'.h5'
    with open(model_file, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weight_file)

    encodingN = model.layers[4].output_shape[1]
    input_1 = Input((None, nfilts*2+1))
    encoded_input = Input(shape=(None,(encodingN+2)))
    decoder_layer2 = model.layers[-3]
    decoder_layer3 = model.layers[-2]
    decoder_layer4 = model.layers[-1]

    merged = model.layers[-4]
    repeater = model.layers[-5]
    encoded2 = model.layers[2]
    encoded1 = model.layers[1]
    encoded = model.layers[3]
    encoded_sig = model.layers[4]
    decoder = Model(encoded_input,decoder_layer4(decoder_layer3(decoder_layer2(encoded_input))))
    encoder = Model(input_1, encoded(encoded2(encoded1(input_1))))
    encoder_sig = Model(input_1, encoded_sig(encoded2(encoded1(input_1))))

    if plot:
        decoder = get_decoder(model, encodingN)
        lms = outseq[:, 0, 1]
        sequence_len = maxlen
        get_decodings(decoder, encoder, sequence, lms, encodingN, sequence_len)

    encodings = np.zeros((len(ids), encodingN))
    for i in np.arange(len(ids)):
        if i%10000==0:
            print(i)
        inseq = np.reshape(sequence[i, :, :], (1, maxlen, nfilts*2+1))
        my_encoding = encoder.predict(inseq)
        encodings[i, :] = my_encoding
        encoder.reset_states()
    return encodings


def feat_peaks(input_lcs):
    """
    Extract peak magnitudes from GP LCs

    Parameters
    ----------
    input_lcs : list
        List of LC objects

    Returns
    -------
    peaks : list
        Peaks from each LC filter

    Examples
    --------
    """
    peaks = []
    for input_lc in input_lcs:
        peaks.append(np.nanmin(input_lc.dense_lc[:, :, 0], axis=0))
    return peaks


def feat_rise_and_decline(input_lcs, n_mag, nfilts=4):

    t_falls_all = []
    t_rises_all = []

    for i, input_lc in enumerate(input_lcs):
        gp = input_lc.gp
        gp_mags = input_lc.gp_mags
        t_falls = []
        t_rises = []
        for j in np.arange(nfilts):
            new_times = np.linspace(-100, 100, 500)
            x_stacked = np.asarray([new_times, [j] * 500]).T
            pred, var = gp.predict(gp_mags, x_stacked)

            max_ind = np.nanargmin(pred)
            max_mag = pred[max_ind]
            max_t = new_times[max_ind]
            trise = np.where((new_times < max_t) & (pred > (max_mag + n_mag)))
            tfall = np.where((new_times > max_t) & (pred > (max_mag + n_mag)))
            if len(trise[0]) == 0:
                trise = np.max(new_times) - max_t
            else:
                trise = max_t - new_times[trise][-1]
            if len(tfall[0]) == 0:
                tfall = max_t - np.min(new_times)
            else:
                tfall = new_times[tfall][0] - max_t

            t_falls.append(tfall)
            t_rises.append(trise)
        t_falls_all.append(t_falls)
        t_rises_all.append(t_rises)
    return t_rises_all, t_falls_all


def feat_slope(input_lcs, t_min_lim=10, t_max_lim=30, nfilts=4):
    slopes_all = []
    for i, input_lc in enumerate(input_lcs):
        gp = input_lc.gp
        gp_mags = input_lc.gp_mags
        slopes = []
        for j in np.arange(nfilts):
            new_times = np.linspace(-100, 100, 500)
            x_stacked = np.asarray([new_times, [j] * 500]).T
            pred, var = gp.predict(gp_mags, x_stacked)
            max_ind = np.nanargmin(pred)
            max_t = new_times[max_ind]
            new_times = new_times - max_t
            lc_grad = np.gradient(pred, new_times)
            gindmean = np.where((new_times > t_min_lim) & (new_times < t_max_lim))
            slopes.append(np.nanmedian(lc_grad[gindmean]))
        slopes_all.append(slopes)
    return slopes_all


def feat_int(input_lcs, nfilts=4):
    ints_all = []
    for i, input_lc in enumerate(input_lcs):
        gp = input_lc.gp
        gp_mags = input_lc.gp_mags
        ints = []
        for j in np.arange(nfilts):
            new_times = np.linspace(-100, 100, 500)
            x_stacked = np.asarray([new_times, [j] * 500]).T
            pred, var = gp.predict(gp_mags, x_stacked)
            ints.append(np.trapz(pred))

        ints_all.append(ints)
    return ints_all


def save_features(features, ids, feat_names, outputfile, outdir):
    # make output dir
    outputfile = outputfile+'.npz'
    outputfile = outdir + outputfile
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    np.savez(outputfile, features=features, ids=ids, feat_names=feat_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lcfile', type=str, help='Light curve file')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data (single file)')
    parser.add_argument('--plot', type=str2bool, default=False, help='Plot LCs, for testing')
    parser.add_argument('--model-base', type=str, dest='model_base', default='./products/models/model', help='...')
    parser.add_argument('--get-feat-raenn', type=str2bool, dest='get_feat_raenn', default=True, help='...')
    parser.add_argument('--get-feat-peaks', type=str2bool, dest='get_feat_peaks', default=False, help='...')
    parser.add_argument('--get-feat-rise-decline-1', type=str2bool,
                        dest='get_feat_rise_decline1', default=False,
                        help='...')
    parser.add_argument('--get-feat-rise-decline-2', type=str2bool,
                        dest='get_feat_rise_decline2', default=False,
                        help='...')
    parser.add_argument('--get-feat-rise-decline-3', type=str2bool,
                        dest='get_feat_rise_decline3', default=False,
                        help='...')
    parser.add_argument('--get-feat-slope', type=str2bool, dest='get_feat_slope', default=False, help='...')
    parser.add_argument('--get-feat-int', type=str2bool, dest='get_feat_int', default=False, help='...')
    parser.add_argument('--prep-file', type=str, dest='prep_file', default='./products/prep.npz', help='...')
    parser.add_argument('--outfile', type=str, dest='outfile', default='feat', help='...')

    args = parser.parse_args()
    features = []

    input_lcs = np.load(args.lcfile, allow_pickle=True)['lcs']
    ids = []
    feat_names = []
    for input_lc in input_lcs:
        if type(input_lc) == float:
            continue
        ids.append(input_lc.name)
    if args.get_feat_raenn:
        feat = feat_from_raenn(args.lcfile, model_base=args.model_base,
                               prep_file=args.prep_file, plot=args.plot)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('raenn'+str(i))
        print('RAENN feat done')

    if args.get_feat_peaks:
        feat = feat_peaks(input_lcs)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('peak'+str(i))
        print('peak feat done')

    if args.get_feat_rise_decline1:
        feat1, feat2 = feat_rise_and_decline(input_lcs, 1)
        if features != []:
            features = np.hstack((features, feat1))
            features = np.hstack((features, feat2))
        else:
            features = np.hstack((feat1, feat2))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('rise1'+str(i))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('decline1'+str(i))
        print('dur1 feat done')

    if args.get_feat_rise_decline2:
        feat1, feat2 = feat_rise_and_decline(input_lcs, 2)
        if features != []:
            features = np.hstack((features, feat1))
            features = np.hstack((features, feat2))
        else:
            features = np.hstack((feat1, feat2))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('rise2'+str(i))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('decline2'+str(i))
        print('dur2 feat done')

    if args.get_feat_rise_decline3:
        feat1, feat2 = feat_rise_and_decline(input_lcs, 3)
        if features != []:
            features = np.hstack((features, feat1))
            features = np.hstack((features, feat2))
        else:
            features = np.hstack((feat1, feat2))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('rise3'+str(i))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('decline3'+str(i))
        print('dur3 feat done')

    if args.get_feat_slope:
        feat = feat_slope(input_lcs)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('slope'+str(i))
        print('slope feat done')

    if args.get_feat_int:
        feat = feat_int(input_lcs)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('int'+str(i))
        print('int feat done')

    if args.outdir[-1] != '/':
        args.outdir += '/'
    save_features(features, ids, feat_names, args.outfile+'_'+date, outdir=args.outdir)
    save_features(features, ids, feat_names, args.outfile, outdir=args.outdir)


if __name__ == '__main__':
    main()
