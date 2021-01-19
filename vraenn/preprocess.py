from argparse import ArgumentParser
import numpy as np
import logging
import datetime
import os
from .lc import LightCurve

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Default fitting parameters
DEFAULT_ZPT = 27#27.5
DEFAULT_LIM_MAG = 26.0


def read_in_LC_files(input_files, obj_names, datastyle='pandas'):
    """
    Read in LC files and convert to LC object

    Parameters
    ----------
    input_files : list
        List of LC file names, to be read in.
    obj_names : list
        List of SNe names, should be same length as input_files
    style : string
        Style of LC files. Assumes SNANA

    Returns
    -------
    lcs : list
        list of Light Curve objects

    Examples
    --------
    """
    LC_list = []
    if datastyle == 'SNANA':
        for i, input_file in enumerate(input_files):
            t, f, filts, err = np.genfromtxt(input_file,
                                             usecols=(1, 4, 2, 5), skip_header=18,
                                             skip_footer=1, unpack=True, dtype=str)
            t = np.asarray(t, dtype=float)
            f = np.asarray(f, dtype=float)
            err = np.asarray(err, dtype=float)

            sn_name = obj_names[i]
            new_LC = LightCurve(sn_name, t, f, err, filts)
            LC_list.append(new_LC)
    elif datastyle == 'text':
        for i, input_file in enumerate(input_files):
            t, f, filts, err, source, upperlim = np.genfromtxt(input_file,
                                                                usecols=(0, 1, 4, 2, 3, 6),
                                                                dtype=str, skip_header=1,
                                                                unpack=True)
            t = np.asarray(t, dtype=float)
            f = np.asarray(f, dtype=float)
            err = np.asarray(err, dtype=float)
            filters = np.asarray(filts, dtype=str)

            f = 10.**(f / -2.5)
            const = np.log(10) / 2.5
            err = err * f / (1.086)

            gind = np.where((source=='ZTF') & (upperlim == 'False'))
            t = t[gind]
            f = f[gind]
            err = err[gind]
            filts = filts[gind]

            sn_name = obj_names[i]
            new_LC = LightCurve(sn_name, t, f, err, filts)
            LC_list.append(new_LC)
    elif datastyle == 'pandas':
        data = np.load(input_files,allow_pickle=True)['lcs']
        for i,sn in enumerate(data):
            t = np.asarray(sn[0],dtype=float)
            f = np.asarray(sn[1],dtype=float)
            err = np.asarray(sn[2],dtype=float)
            filts = np.asarray(sn[3],dtype=str)
            sn_name = obj_names[i]
            new_LC = LightCurve(sn_name, t, f, err, filts)
            LC_list.append(new_LC)
    else:
        raise ValueError('Sorry, you need to specify a data style.')
    return LC_list

def read_in_meta_table(metatable):
    """
    Read in the metatable file

    Parameters
    ----------
    metatable : str
        Name of metatable file
    Returns
    -------
    obj : numpy.ndarray
        Array of object IDs (strings)
    redshift : numpy.ndarray
        Array of redshifts
    redshift_err : numpy.ndarray
        Array of redshift errors
    obj_type : numpy.ndarray
        Array of SN spectroscopic types
    my_peak : numpy.ndarray
        Array of best-guess peak times
    ebv : numpy.ndarray
        Array of MW ebv values

    Todo
    ----------
    Make metatable more flexible
    """
    obj, redshift, redshift_err, obj_type, \
        my_peak, ebv = np.loadtxt(metatable, unpack=True, dtype=str, delimiter=' ')
    redshift = np.asarray(redshift, dtype=float)
    my_peak = np.asarray(my_peak, dtype=float)
    ebv = np.asarray(ebv, dtype=float)

    return obj, redshift, redshift_err, obj_type, my_peak, ebv


def save_lcs(lc_list, output_dir):
    """
    Save light curves as a lightcurve object

    Parameters
    ----------
    lc_list : list
        list of light curve files
    output_dir : Output directory of light curve file

    Todo:
    ----------
    - Add option for LC file name
    """
    now = datetime.datetime.now()
    date = str(now.strftime("%Y-%m-%d"))
    file_name = 'lcs_' + date + '.npz'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_dir[-1] != '/':
        output_dir += '/'

    output_file = output_dir + file_name
    np.savez(output_file, lcs=lc_list)
    # Also easy save to latest
    np.savez(output_dir+'lcs.npz', lcs=lc_list)

    logging.info(f'Saved to {output_file}')


def main():
    """
    Preprocess the LC files
    """

    # Create argument parser
    parser = ArgumentParser()
    parser.add_argument('datadir', type=str, help='Directory of LC files')
    parser.add_argument('metatable', type=str,
                        help='Metatable containing each object, redshift, peak time guess, mwebv, object type')
    parser.add_argument('--zpt', type=float, default=DEFAULT_ZPT, help='Zero point of LCs')
    parser.add_argument('--lm', type=float, default=DEFAULT_LIM_MAG, help='Survey limiting magnitude')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data (single file)')
    parser.add_argument('--datatype', type=str, default = 'ZTF', help='LSST (PS1) or ZTF')
    parser.add_argument('--datastyle', type=str, default = 'text', help='SNANA or text')
    parser.add_argument('--shifttype', type=str, default = 'peak', help='how to shift time. Input time or peak')
    args = parser.parse_args()

    objs, redshifts, redshift_errs, obj_types, peaks, ebvs = read_in_meta_table(args.metatable)

    # Grab all the LC files in the input directory
    file_names = []
    for obj in objs:
        if args.datastyle == 'SNANA':
            file_name = args.datadir + 'PS1_PS1MD_' + obj + '.snana.dat'
        else:
            file_name = args.datadir + obj + '.txt'

        file_names.append(file_name)

    if args.datastyle == 'pandas':
        file_names = args.datadir

    # Create a list of LC objects from the data files
    lc_list = read_in_LC_files(file_names, objs, datastyle=args.datastyle)

    # This needs to be redone when retrained
    if args.datatype == 'PS1':
        filt_dict = {'g': 0, 'r': 1, 'i': 2, 'z': 3}
        wvs = np.asarray([5460, 6800, 7450, 8700])
        nfilt = 4
    elif args.datatype == 'LSST':
        filt_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4':4, '5':5}
        wvs = np.asarray([3740, 4870, 6250, 7700, 8900, 10845])
        nfilt = 6
    else:
        filt_dict = {'g': 0, 'r': 1}
        wvs = np.asarray([5460, 6800])
        nfilt = 2

    # Update the LC objects with info from the metatable
    my_lcs = [np.nan] * len(lc_list)
    for i, my_lc in enumerate(lc_list):
        print(i)
        my_lc.add_LC_info(zpt=args.zpt, mwebv=ebvs[i],
                          redshift=redshifts[i], redshift_err = redshift_errs[i], lim_mag=args.lm,
                          obj_type=obj_types[i])
        my_lc.get_abs_mags()

        if np.inf in my_lc.abs_mags:
            continue

        my_lc.sort_lc()
        if my_lc.times.size < 1:
            continue


        if args.shifttype == 'peak':
            pmjd = my_lc.find_peak(peaks[i])
        else:
            pmjd = peaks[i]
        my_lc.shift_lc(pmjd)
        my_lc.correct_time_dilation()
        my_lc.filter_names_to_numbers(filt_dict)
        my_lc.correct_extinction(wvs)
        my_lc.cut_lc()
        if my_lc.times.size < 3:
            continue
        my_lc.make_dense_LC(nfilt)

        my_lcs[i] = my_lc
    save_lcs(my_lcs, args.outdir)


if __name__ == '__main__':
    main()
